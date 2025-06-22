import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dotenv import load_dotenv
import json
import pickle

from config import ISLConfig
from pytorchDatapreprocessor import DataPreprocessor
from pytorchModelBuilder import ISLModelBuilder
from pytorchTrainer import ISLTrainer
from pytorchEvaluator import ISLEvaluator
from logger import setup_logger
import os
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'

load_dotenv()

def main():
    logger = setup_logger("pytorch_pipeline")
    
    # Set device


    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    print("Using device", device )
    
    try:
        logger.info("Step 1: Initializing configuration")
        config = ISLConfig()
        
        
        logger.info("Step 2: Loading and preprocessing data")
        preprocessor = DataPreprocessor(config)

        manifest, class_names =preprocessor.load_and_preprocess_data()
        num_classes = len(class_names)
        config.NUM_CLASSES = num_classes
        print(f"Number of classes: {config.NUM_CLASSES}")
        
        # Load data for training
        train_loader, test_loader = preprocessor.load_data_for_training(
            batch_size=config.BATCH_SIZE,
            num_workers=4,
            shuffle=True
        )
        
        # Test data loading
        logger.info("Testing data loading...")
        for batch_idx, (data, target) in enumerate(train_loader):
            frames, landmarks = data
            print(f"Batch {batch_idx}:")
            print(f"  Frames shape: {frames.shape}")
            print(f"  Landmarks shape: {landmarks.shape}")
            print(f"  Labels shape: {target.shape}")
            print(f"  Device: {frames.device}")
            break
        
        logger.info("Step 3: Building the model")
        model_builder = ISLModelBuilder(config)
        model = model_builder.create_model(num_classes=config.NUM_CLASSES)
        model = model.to(device)
        
        # Test model with a batch
        logger.info("Testing model forward pass...")
        with torch.no_grad():
            frames, landmarks = frames.to(device), landmarks.to(device)
            output = model(frames, landmarks)
            print(f"Model output shape: {output.shape}")
        
        logger.info("Step 4: Setting up trainer")
        trainer = ISLTrainer(
            model=model,
            config=config,
            device=device,
            
        )
        
        logger.info("Step 5: Starting training")
        optimizer = model_builder.get_optimizer(model)
        num_training_steps = config.EPOCHS * len(train_loader)
        scheduler = model_builder.get_scheduler(optimizer, num_training_steps)

        trainer.train(train_loader=train_loader,
            val_loader=test_loader,optimizer=optimizer,scheduler=scheduler)
        
        logger.info("Step 6: Evaluating model")
        evaluator = ISLEvaluator(model, test_loader, device, class_names)
        test_accuracy, test_loss = evaluator.evaluate()
        
        logger.info(f"Final test accuracy: {test_accuracy:.4f}")
        logger.info(f"Final test loss: {test_loss:.4f}")
        
        # Generate detailed evaluation report
        evaluator.generate_report()
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()