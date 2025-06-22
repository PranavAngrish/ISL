import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import logging
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report
from logger import setup_logger
import gc

class ISLTrainer:
    """Memory-optimized PyTorch trainer for ISL model"""
    
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        
        # Use provided device or auto-detect
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Trainer using device: {self.device}")
        
        # MPS memory optimization
        if self.device.type == 'mps':
            # Enable MPS memory management
            torch.mps.empty_cache()
            print("MPS cache cleared")
            
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Mixed precision training - disable for MPS
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            print("Mixed precision disabled for non-CUDA device")
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_top3_acc': []
        }
        
        # Setup logging
        self.logger = setup_logger(name="ISLTrainer")
        
        # Memory optimization settings
        self.gradient_accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        self.memory_cleanup_frequency = 10  # Clean memory every N batches
        
    def cleanup_memory(self):
        """Clean up GPU/MPS memory"""
        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()
        elif self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        
    


    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """Memory-optimized training for one epoch with OOM batch retry"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc='Training')

        for batch_idx, ((frames, landmarks), labels) in enumerate(pbar):
            retry = True
            while retry:
                try:
                    # Debug: Check input device (only for first batch)
                    if batch_idx == 0:
                        print(f"Input frames shape: {frames.shape}")
                        print(f"Input landmarks shape: {landmarks.shape}")
                        print(f"Batch size: {frames.size(0)}")
                        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

                    # Move to device
                    frames = frames.to(self.device, non_blocking=True)
                    landmarks = landmarks.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # Forward pass
                    if self.use_amp:
                        with autocast(device_type=self.device.type):
                            logits = self.model(frames, landmarks)
                            loss = self.criterion(logits, labels)
                    else:
                        logits = self.model(frames, landmarks)
                        loss = self.criterion(logits, labels)

                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                    # Backward pass
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Gradient accumulation step
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.use_amp:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        if scheduler:
                            scheduler.step()

                    # Calculate accuracy
                    with torch.no_grad():
                        _, predicted = torch.max(logits.data, 1)
                        total_samples += labels.size(0)
                        total_correct += (predicted == labels).sum().item()
                        total_loss += loss.item() * self.gradient_accumulation_steps

                    # Memory cleanup
                    if batch_idx % self.memory_cleanup_frequency == 0:
                        self.cleanup_memory()

                    # Update progress bar
                    current_acc = 100.0 * total_correct / total_samples
                    pbar.set_postfix({
                        'Loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                        'Acc': f'{current_acc:.2f}%'
                    })

                    # Clear variables to free memory
                    del frames, landmarks, labels, logits

                    retry = False  # Success, exit retry loop

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error at batch {batch_idx}, cleaning up memory and retrying...")
                        self.cleanup_memory()
                        # Optionally, add a retry limit to avoid infinite loops
                    else:
                        raise e

        # Final gradient step if needed
        if len(train_loader) % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * total_correct / total_samples

        return epoch_loss, epoch_acc




    def validate_epoch(self, val_loader):
        """Memory-optimized validation for one epoch with OOM batch retry"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')

            for batch_idx, ((frames, landmarks), labels) in enumerate(pbar):
                retry = True
                while retry:
                    try:
                        # Move to device
                        frames = frames.to(self.device, non_blocking=True)
                        landmarks = landmarks.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                        # Forward pass
                        if self.use_amp:
                            with autocast(device_type=self.device.type):
                                logits = self.model(frames, landmarks)
                                loss = self.criterion(logits, labels)
                        else:
                            logits = self.model(frames, landmarks)
                            loss = self.criterion(logits, labels)

                        # Calculate accuracy
                        _, predicted = torch.max(logits.data, 1)
                        total_samples += labels.size(0)
                        total_correct += (predicted == labels).sum().item()
                        total_loss += loss.item()

                        # Store predictions for top-k accuracy (move to CPU immediately)
                        all_predictions.append(logits.cpu())
                        all_labels.append(labels.cpu())

                        # Memory cleanup
                        if batch_idx % self.memory_cleanup_frequency == 0:
                            self.cleanup_memory()

                        # Update progress bar
                        current_acc = 100.0 * total_correct / total_samples
                        pbar.set_postfix({
                            'Loss': f'{loss.item():.4f}',
                            'Acc': f'{current_acc:.2f}%'
                        })

                        # Clear variables to free memory
                        del frames, landmarks, labels, logits

                        retry = False  # Success, exit retry loop

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"OOM error during validation at batch {batch_idx}, cleaning up and retrying...")
                            self.cleanup_memory()
                            # Optionally, add a retry limit here to avoid infinite loops
                        else:
                            raise e

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100.0 * total_correct / total_samples

        # Calculate top-3 accuracy
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Convert to numpy for sklearn
            pred_probs = F.softmax(all_predictions, dim=1).numpy()
            labels_np = all_labels.numpy()

            top3_acc = 100.0 * top_k_accuracy_score(labels_np, pred_probs, k=3)
        else:
            top3_acc = 0.0

        return epoch_loss, epoch_acc, top3_acc




    
    def save_checkpoint(self, epoch, optimizer, scheduler=None, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config.__dict__,
            'device': str(self.device)
        }
        
        # Create checkpoint directory
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR, 
            f'checkpoint_epoch_{epoch:02d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with val_acc: {self.best_val_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_top3_acc': []
        })
        
        epoch = checkpoint['epoch']
        self.logger.info(f"Checkpoint loaded from epoch {epoch}")
        return epoch
    
    def train(self, train_loader, val_loader, optimizer, scheduler=None, start_epoch=0):
        """Main training loop with memory optimization"""
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Using mixed precision: {self.use_amp}")
        self.logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # Initial memory cleanup
        self.cleanup_memory()
        
        # Early stopping parameters
         
        patience = self.config.EARLY_STOPPING_PATIENCE
        patience_counter = 0
        
        for epoch in range(start_epoch, self.config.EPOCHS):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.EPOCHS}")
            
            # Clean memory before each epoch
            self.cleanup_memory()
            
            try:
                # Training phase
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
                
                # Validation phase
                val_loss, val_acc, val_top3_acc = self.validate_epoch(val_loader)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_top3_acc'].append(val_top3_acc)
                
                # Print epoch results
                self.logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Top3 Acc: {val_top3_acc:.2f}%")
                
                # Check for best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Save checkpoint
                self.save_checkpoint(epoch, optimizer, scheduler, is_best)
                print("Checkpoint saved")
                
                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.error(f"Out of memory error in epoch {epoch+1}. Try reducing batch size or enabling gradient accumulation.")
                    print(f"Out of memory error in epoch {epoch+1}. Try reducing batch size.")
                    break
                else:
                    raise e
        
        self.logger.info(f"Training completed! Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"Training completed! Best Val Acc: {self.best_val_acc:.2f}%")
        return self.history
    
    def evaluate(self, test_loader, class_names=None):
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, ((frames, landmarks), labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
                try:
                    frames = frames.to(self.device)
                    landmarks = landmarks.to(self.device)
                    
                    logits = self.model(frames, landmarks)
                    _, predicted = torch.max(logits.data, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    
                    # Memory cleanup
                    if batch_idx % self.memory_cleanup_frequency == 0:
                        self.cleanup_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM during evaluation at batch {batch_idx}, skipping...")
                        self.cleanup_memory()
                        continue
                    else:
                        raise e
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Classification report
        if class_names:
            report = classification_report(
                all_labels, all_predictions, 
                target_names=class_names, 
                output_dict=True
            )
        else:
            report = classification_report(all_labels, all_predictions, output_dict=True)
        
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels
        }