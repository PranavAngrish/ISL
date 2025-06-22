import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import numpy as np
from config import ISLConfig
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

class TemporalFusionLayer(nn.Module):
    """Temporal fusion layer using transformer attention"""
    
    def __init__(self, embed_dim, sequence_length, dropout_rate=0.1, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        
        # Temporal positional embeddings
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, sequence_length, embed_dim) * 0.02
        )
        
        # Multi-head attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        # x shape: (B, T, D)
        x = x + self.temporal_pos_embed
        
        # Self-attention
        attn_output, _ = self.temporal_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        return x

class LandmarkProcessor(nn.Module):
    """Process landmark features"""
    
    def __init__(self, embed_dim, sequence_length, dropout_rate=0.1, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Landmark embedding
        self.landmark_embedding = nn.Linear(170, embed_dim)
        self.landmark_dropout = nn.Dropout(dropout_rate)
        
        # Positional embeddings
        self.landmark_pos_embed = nn.Parameter(
            torch.randn(1, sequence_length, embed_dim) * 0.02
        )
        
        # Attention
        self.landmark_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, landmarks):
        # landmarks shape: (B, T, 170)
        x = self.landmark_embedding(landmarks)
        x = self.landmark_dropout(x)
        x = x + self.landmark_pos_embed
        
        # Self-attention
        attn_output, _ = self.landmark_attention(x, x, x)
        x = self.layer_norm(x + attn_output)
        
        return x

class CrossModalFusion(nn.Module):
    """Cross-modal fusion between video and landmark features"""
    
    def __init__(self, embed_dim, dropout_rate=0.1, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Cross-attention layers
        self.video_to_landmark_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.landmark_to_video_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, video_features, landmark_features):
        # Cross-attention: video -> landmark
        video_attended, _ = self.video_to_landmark_attention(
            query=video_features,
            key=landmark_features,
            value=landmark_features
        )
        video_attended = self.layer_norm1(video_features + video_attended)
        
        # Cross-attention: landmark -> video
        landmark_attended, _ = self.landmark_to_video_attention(
            query=landmark_features,
            key=video_features,
            value=video_features
        )
        landmark_attended = self.layer_norm2(landmark_features + landmark_attended)
        
        # Fusion
        fused = self.fusion_layer(video_attended + landmark_attended)
        return fused

class ISLViTModel(nn.Module):
    """Complete ISL detection model with ViT backbone"""
    
    def __init__(self, config: ISLConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Load pre-trained ViT model
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # Freeze early layers (optional)
        for name, param in self.vit_model.named_parameters():
            if 'encoder.layer' in name:
                layer_num = int(name.split('.')[2])
                if layer_num < 8:  # Freeze first 8 layers
                    param.requires_grad = False
        
        # Get ViT embedding dimension
        self.vit_embed_dim = self.vit_model.config.hidden_size  # 768 for base model
        
        # Project ViT features to our embedding dimension
        self.vit_projection = nn.Linear(self.vit_embed_dim, config.EMBED_DIM)
        
        # Temporal fusion
        self.temporal_fusion = TemporalFusionLayer(
            embed_dim=config.EMBED_DIM,
            sequence_length=config.SEQUENCE_LENGTH,
            dropout_rate=config.DROPOUT_RATE,
            num_heads=config.NUM_HEADS
        )
        
        # Landmark processor
        self.landmark_processor = LandmarkProcessor(
            embed_dim=config.EMBED_DIM,
            sequence_length=config.SEQUENCE_LENGTH,
            dropout_rate=config.DROPOUT_RATE,
            num_heads=config.NUM_HEADS // 2
        )
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(
            embed_dim=config.EMBED_DIM,
            dropout_rate=config.DROPOUT_RATE,
            num_heads=config.NUM_HEADS
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.EMBED_DIM, config.EMBED_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.EMBED_DIM, config.EMBED_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.EMBED_DIM // 2, num_classes)
        )
        
    def forward(self, frames, landmarks):
        """
        Forward pass
        Args:
            frames: (B, T, C, H, W) - video frames
            landmarks: (B, T, 170) - landmark features
        """
        batch_size, sequence_length = frames.shape[:2]
        
        # Reshape frames for ViT: (B*T, C, H, W)
        frames_reshaped = frames.view(batch_size * sequence_length, *frames.shape[2:])
        
        # Process through ViT
        with torch.cuda.amp.autocast():
            vit_outputs = self.vit_model(pixel_values=frames_reshaped)
            # Get CLS token features: (B*T, hidden_size)
            frame_features = vit_outputs.last_hidden_state[:, 0, :]
        
        # Project to our embedding dimension
        frame_features = self.vit_projection(frame_features)
        
        # Reshape back to sequence: (B, T, embed_dim)
        frame_features = frame_features.view(batch_size, sequence_length, self.config.EMBED_DIM)
        
        # Temporal fusion
        video_features = self.temporal_fusion(frame_features)
        
        # Process landmarks
        landmark_features = self.landmark_processor(landmarks)
        
        # Cross-modal fusion
        fused_features = self.cross_modal_fusion(video_features, landmark_features)
        
        # Global average pooling over time dimension
        pooled_features = fused_features.mean(dim=1)  # (B, embed_dim)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits

class ISLModelBuilder:
    """Model builder using pre-trained Vision Transformers from Hugging Face"""

    def __init__(self, config: ISLConfig):
        self.config = config

    def create_model(self, num_classes: int):
        """Create the complete ISL model"""
        print("Building PyTorch model with pre-trained ViT backbone...")
        
        model = ISLViTModel(self.config, num_classes)
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def get_optimizer(self, model):
        """Get optimizer for the model"""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def get_scheduler(self, optimizer, num_training_steps):
        """Get learning rate scheduler"""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)