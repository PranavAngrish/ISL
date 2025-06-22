import os
import cv2
import numpy as np
import mediapipe as mp
import json
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from config import ISLConfig
import torch
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    """PyTorch Dataset for video data"""
    
    def __init__(self, manifest, transform=None):
        self.manifest = manifest
        self.transform = transform
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        entry = self.manifest[idx]
        
        try:
            # Load frames and landmarks
            frames = np.load(entry["frame_path"])  # Shape: (T, H, W, C)
            landmarks = np.load(entry["landmarks_path"])  # Shape: (T, 170)
            label = entry["encoded_label"]
            
            # Convert to torch tensors
            frames = torch.from_numpy(frames).float()
            landmarks = torch.from_numpy(landmarks).float()
            label = torch.tensor(label, dtype=torch.long)
            
            # Permute frames to (T, C, H, W) for PyTorch
            frames = frames.permute(0, 3, 1, 2)
            
            if self.transform:
                frames, landmarks = self.transform(frames, landmarks)
            
            return (frames, landmarks), label
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample in case of error
            T, H, W, C = 16, 224, 224, 3
            frames = torch.zeros(T, C, H, W)
            landmarks = torch.zeros(T, 170)
            label = torch.tensor(0, dtype=torch.long)
            return (frames, landmarks), label

class DataPreprocessor:
    """Data preprocessing module for ISL detection using PyTorch"""
    
    def __init__(self, config: ISLConfig):
        self.config = config
        self.config.create_directories()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        self.label_encoder = LabelEncoder()
        
    def extract_landmarks(self, frame):
        """Extract hand and pose landmarks from a frame using MediaPipe"""
        landmarks = []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Hand landmarks
        hand_results = self.hands.process(rgb_frame)
        hand_landmarks_list = [np.zeros(63), np.zeros(63)]  # Prepare 2 hand slots

        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                if i >= 2:  # We only support MAX_NUM_HANDS = 2
                    break
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                hand_landmarks_list[i] = np.array(coords)

        # Add both hands' landmarks
        landmarks.extend(hand_landmarks_list[0])
        landmarks.extend(hand_landmarks_list[1])
        
        # Pose landmarks (upper body only)
        pose_results = self.pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            # Extract only upper body landmarks (0-10)
            for i in range(11):
                landmark = pose_results.pose_landmarks.landmark[i]
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            landmarks.extend([0.0] * 44)  # 11 landmarks * 4 coordinates
        
        return np.array(landmarks)
    
    def preprocess_video(self, video_path):
        """Preprocess a single video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        landmarks_sequence = []
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract
        if total_frames > self.config.SEQUENCE_LENGTH:
            frame_indices = np.linspace(0, total_frames - 1, self.config.SEQUENCE_LENGTH, dtype=int)
        else:
            frame_indices = list(range(total_frames))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_indices:
                # Resize frame
                frame_resized = cv2.resize(frame, self.config.IMG_SIZE)
                frames.append(frame_resized)
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                landmarks_sequence.append(landmarks)
            
            frame_idx += 1
        
        cap.release()
        
        # Pad sequences if needed
        while len(frames) < self.config.SEQUENCE_LENGTH:
            if frames:
                frames.append(frames[-1])  # Repeat last frame
                landmarks_sequence.append(landmarks_sequence[-1])  # Repeat last landmarks
            else:
                # Create blank frame if no frames extracted
                blank_frame = np.zeros((*self.config.IMG_SIZE, 3), dtype=np.uint8)
                frames.append(blank_frame)
                landmarks_sequence.append(np.zeros(170))  # 63 + 63 + 44 landmarks
        
        return np.array(frames[:self.config.SEQUENCE_LENGTH]), np.array(landmarks_sequence[:self.config.SEQUENCE_LENGTH])

    def load_and_preprocess_data(self):
        """Process and save video data, preserving input folder structure."""
        print("Starting data preprocessing with preserved folder structure...")

        manifest = []
        label_set = set()

        categories = [d for d in os.listdir(self.config.DATA_PATH)
                      if os.path.isdir(os.path.join(self.config.DATA_PATH, d))]

        sample_index = 0
        for category in tqdm(categories, desc="Processing categories"):
            category_path = os.path.join(self.config.DATA_PATH, category)

            classes = [d for d in os.listdir(category_path)
                       if os.path.isdir(os.path.join(category_path, d))]

            for class_name in tqdm(classes, desc=f"Processing {category}", leave=False):
                class_path = os.path.join(category_path, class_name)

                video_files = [f for f in os.listdir(class_path)
                               if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

                for video_file in video_files:
                    video_path = os.path.join(class_path, video_file)

                    try:
                        frames, landmarks = self.preprocess_video(video_path)

                        # Create output directory matching the input structure
                        video_name = os.path.splitext(video_file)[0]
                        target_dir = os.path.join(self.config.CHUNKED_DATA_DIR, category, class_name, video_name)
                        os.makedirs(target_dir, exist_ok=True)

                        # Save frames and landmarks (normalized frames)
                        frame_path = os.path.join(target_dir, "frames.npy")
                        landmarks_path = os.path.join(target_dir, "landmarks.npy")
                        np.save(frame_path, frames.astype('float32') / 255.0)
                        np.save(landmarks_path, landmarks)

                        # Metadata
                        label = f"{category}_{class_name}"
                        label_set.add(label)

                        sample_meta = {
                            "index": sample_index,
                            "label": label,
                            "original_video": video_path,
                            "frame_path": frame_path,
                            "landmarks_path": landmarks_path,
                            "sequence_length": self.config.SEQUENCE_LENGTH,
                            "frame_shape": list(frames.shape),
                            "landmarks_shape": list(landmarks.shape)
                        }

                        with open(os.path.join(target_dir, "metadata.json"), 'w') as f:
                            json.dump(sample_meta, f, indent=2)

                        manifest.append(sample_meta)
                        sample_index += 1
    
                    except Exception as e:
                        print(f"❌ Error processing {video_path}: {e}")
                        continue

        # Encode labels
        labels = [entry["label"] for entry in manifest]
        y_encoded = self.label_encoder.fit_transform(labels)
        class_names = list(self.label_encoder.classes_)

        # Add encoded labels to each entry + update each metadata.json
        for i, entry in enumerate(manifest):
            entry["encoded_label"] = int(y_encoded[i])

            metadata_path = os.path.join(self.config.CHUNKED_DATA_DIR,
                                         os.path.relpath(entry["frame_path"], start=self.config.CHUNKED_DATA_DIR))
            metadata_path = os.path.join(os.path.dirname(metadata_path), "metadata.json")

            with open(metadata_path, 'w') as f:
                json.dump(entry, f, indent=2)

        # Save manifest and label encoder
        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)

        with open(os.path.join(self.config.CHUNKED_DATA_DIR, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"✅ Saved {sample_index} samples in structured format.")
        print(f"🔖 Found {len(class_names)} classes.")
        return manifest, class_names

    @staticmethod
    def normalize_transform(frames, landmarks):
        """Normalization transform for PyTorch tensors"""
        # Frames are already normalized during preprocessing
        return frames, landmarks

    def load_data_for_training(self, test_size=0.2, batch_size=32, random_state=42, num_workers=2, shuffle=True, verbose=True):
        """
        Load data for training using PyTorch DataLoaders.
        """
        if verbose:
            print("=" * 60)
            print("PHASE 2: LOADING DATA FOR TRAINING (PyTorch DataLoaders)")
            print("=" * 60)

        # Load manifest
        manifest_path = os.path.join(self.config.CHUNKED_DATA_DIR, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Load label encoder
        label_encoder_path = os.path.join(self.config.CHUNKED_DATA_DIR, "label_encoder.pkl")
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}")

        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

        if verbose:
            print(f"📋 Found {len(manifest)} samples")
            print(f"🔖 Found {len(self.label_encoder.classes_)} classes")
            print(f"📁 Loading from: {self.config.CHUNKED_DATA_DIR}")

        # Stratified split based on encoded_label
        labels = [entry["encoded_label"] for entry in manifest]
        manifest_train, manifest_test = train_test_split(
            manifest,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        if verbose:
            print(f"✅ Train/Test split completed:")
            print(f"   Train samples: {len(manifest_train)}")
            print(f"   Test samples: {len(manifest_test)}")
            print("=" * 60)
            print("PHASE 2 COMPLETED SUCCESSFULLY!")
            print("=" * 60)

        # Create PyTorch datasets
        train_dataset = VideoDataset(manifest_train, transform=self.normalize_transform)
        test_dataset = VideoDataset(manifest_test, transform=self.normalize_transform)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, test_loader

    def save_processed_data(self, X_frames, X_landmarks, y_encoded, class_names):
        """Save processed data to disk"""
        print("Saving processed data...")
        
        processed_files = self.config.get_processed_files_paths()
        
        # Save arrays
        np.save(processed_files['frames'], X_frames)
        np.save(processed_files['landmarks'], X_landmarks)
        np.save(processed_files['labels'], y_encoded)
        np.save(processed_files['class_names'], class_names)
        
        # Save label encoder
        with open(os.path.join(self.config.PROCESSED_DATA_DIR, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(X_frames),
            'num_classes': len(class_names),
            'frames_shape': X_frames.shape,
            'landmarks_shape': X_landmarks.shape,
            'sequence_length': self.config.SEQUENCE_LENGTH,
            'img_size': self.config.IMG_SIZE
        }
        
        with open(processed_files['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Processed data saved successfully!")
    
    def load_processed_data(self):
        """Load previously processed data"""
        print("Loading processed data...")
        
        processed_files = self.config.get_processed_files_paths()
        
        X_frames = np.load(processed_files['frames'])
        X_landmarks = np.load(processed_files['landmarks'])
        y_encoded = np.load(processed_files['labels'])
        class_names = np.load(processed_files['class_names'])
        
        # Load label encoder
        with open(os.path.join(self.config.PROCESSED_DATA_DIR, "label_encoder.pkl"), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"Loaded: {len(X_frames)} samples, {len(class_names)} classes")
        
        return X_frames, X_landmarks, y_encoded, class_names
    
    def get_data_info(self):
        """Get information about processed data"""
        processed_files = self.config.get_processed_files_paths()
        
        if os.path.exists(processed_files['metadata']):
            with open(processed_files['metadata'], 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            return None