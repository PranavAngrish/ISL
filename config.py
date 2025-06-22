import os
from dataclasses import dataclass
from typing import Tuple
import json
from sklearn.preprocessing import LabelEncoder

@dataclass
class ISLConfig:
    """Configuration class for ISL detection pipeline"""
    
    # Data paths
    DATA_PATH: str = "Data"
    PROCESSED_DATA_DIR: str = "processed_data"
    MODEL_DIR: str = "models"
    RESULTS_DIR: str = "results"
    LOGS_DIR: str = "logs"
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Video processing parameters
    IMG_SIZE: Tuple[int, int] = (224, 224)
    SEQUENCE_LENGTH: int = 16
    BATCH_SIZE: int = 4
    
    # Model parameters
    PRETRAINED_MODEL_NAME: str = "google/vit-base-patch16-224"
    PRETRAINED_VIDEO_MODEL_NAME: str = "MCG-NJU/videomae-base"  # Alternative: "microsoft/xclip-base-patch32"
    NUM_TRANSFORMER_LAYERS: int = 4
    EMBED_DIM: int = 768
    NUM_HEADS: int = 8
    DROPOUT_RATE: float = 0.1
    
    # Training parameters
    LEARNING_RATE: float = 1e-5
    WEIGHT_DECAY: float = 0.05
    EPOCHS: int = 100
    PATIENCE: int = 15
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.2
    WARMUP_STEPS: int = 1000
    EARLY_STOPPING_PATIENCE = 10

    #Scheduler
    SCHEDULER_TYPE = "cosine"  # "cosine", "linear", "polynomial"
    SCHEDULER_WARMUP_RATIO = 0.1
    MIN_LR = 1e-6

    # Optimizer
    OPTIMIZER_TYPE = "adamw"  # "adamw", "adam", "sgd"
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    
    # MediaPipe parameters
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    MAX_NUM_HANDS: int = 2

        # Validation and testing
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    RANDOM_SEED = 42
        
        # Hardware configuration
    NUM_WORKERS = 0
    PIN_MEMORY = False
    PERSISTENT_WORKERS = True
        
        # Checkpointing
    SAVE_EVERY_N_EPOCHS = 5
    SAVE_BEST_ONLY = True
    MONITOR_METRIC = "val_accuracy"
    MONITOR_MODE = "max"
    CHECKPOINT_DIR = "checkpoints"
        
    # Logging
    LOG_LEVEL = "INFO"
    LOG_EVERY_N_STEPS = 100
    SAVE_PREDICTIONS = True
        
    # Mixed precision training
    USE_MIXED_PRECISION = True
    GRADIENT_CLIP_VAL = 1.0
        
    # Model architecture specific
    FREEZE_BACKBONE_LAYERS = 8  # Number of ViT layers to freeze
    USE_PRETRAINED_WEIGHTS = True
        
    # Will be set dynamically
    NUM_CLASSES = None
    CLASS_NAMES = None    

    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.PROCESSED_DATA_DIR,
            self.MODEL_DIR,
            self.RESULTS_DIR,
            self.LOGS_DIR,
            os.path.join(self.PROCESSED_DATA_DIR, "frames"),
            os.path.join(self.PROCESSED_DATA_DIR, "landmarks"),
            os.path.join(self.PROCESSED_DATA_DIR, "labels")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_processed_files_paths(self):
        """Get paths for processed data files"""
        return {
            'frames': os.path.join(self.PROCESSED_DATA_DIR, "frames", "frames.npy"),
            'landmarks': os.path.join(self.PROCESSED_DATA_DIR, "landmarks", "landmarks.npy"),
            'labels': os.path.join(self.PROCESSED_DATA_DIR, "labels", "labels.npy"),
            'class_names': os.path.join(self.PROCESSED_DATA_DIR, "labels", "class_names.npy"),
            'metadata': os.path.join(self.PROCESSED_DATA_DIR, "metadata.json")
        }
    
    @property
    def CHUNKED_DATA_DIR(self):
        return os.path.join(self.PROCESSED_DATA_DIR, "chunked")
    
    def extract_num_classes(manifest_path):
        """Extract number of unique classes from manifest file"""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Extract all labels
        labels = [entry["label"] for entry in manifest]

        # Use LabelEncoder to get encoded labels and classes
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        class_names = list(label_encoder.classes_)
        
        num_classes = len(class_names)

        print(f"Found {num_classes} unique classes:")
        print(class_names)

        return num_classes, class_names, label_encoder