
"""
MAET (Multimodal Attention-Enhanced Transformer) Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .multiview_embedding import MultiViewEmbedding
from .attention_blocks import TransformerBlock
from ..utils.gradient_reversal import gradient_reversal


class MAET(nn.Module):
    """
    Multimodal Attention-Enhanced Transformer for EEG-eye emotion recognition
    
    Args:
        eeg_dim (int): EEG feature dimension (default: 310)
        eye_dim (int): Eye tracking feature dimension (default: 33)
        num_classes (int): Number of emotion classes (default: 7)
        embed_dim (int): Embedding dimension (default: 32)
        depth (int): Number of transformer layers (default: 3)
        num_heads (int): Number of attention heads (default: 4)
        domain_generalization (bool): Enable domain adversarial training
        num_domains (int): Number of source domains for domain adaptation
    """
    
    def __init__(self, eeg_dim=310, eye_dim=33, num_classes=7, embed_dim=32, 
                 depth=3, num_heads=4, domain_generalization=False, num_domains=20):
        super().__init__()
        
        self.eeg_dim = eeg_dim
        self.eye_dim = eye_dim
        self.embed_dim = embed_dim
        self.domain_generalization = domain_generalization
        
        # Embedding modules for different modalities
        if eeg_dim > 0:
            self.eeg_transform = MultiViewEmbedding(eeg_dim, embed_dim)
            self.eeg_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if eye_dim > 0:
            self.eye_transform = MultiViewEmbedding(eye_dim, embed_dim)
            self.eye_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification heads
        self.head = nn.Linear(embed_dim, num_classes)
        if eeg_dim > 0:
            self.head_eeg = nn.Linear(embed_dim, num_classes)
        if eye_dim > 0:
            self.head_eye = nn.Linear(embed_dim, num_classes)
        
        # Domain adversarial classifier for generalization
        if domain_generalization:
            self.domain_classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), 
                nn.GELU(),
                nn.Linear(embed_dim, num_domains)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using truncated normal distribution"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, eeg=None, eye=None, alpha_=1.0):
        """
        Forward pass through MAET model
        
        Args:
            eeg (torch.Tensor): EEG features [batch_size, eeg_dim]
            eye (torch.Tensor): Eye tracking features [batch_size, eye_dim] 
            alpha_ (float): Gradient reversal strength for domain adaptation
            
        Returns:
            torch.Tensor or tuple: Emotion predictions, optionally with domain predictions
        """
        batch_size = eeg.size(0) if eeg is not None else eye.size(0)
        features = []
        
        # Process EEG modality
        if eeg is not None:
            eeg_views = self.eeg_transform(eeg)  # Multi-view embedding
            eeg_cls = self.eeg_cls_token.expand(batch_size, -1, -1)
            eeg_tokens = torch.cat([eeg_cls, eeg_views], dim=1)
            features.append(eeg_tokens)
        
        # Process eye tracking modality  
        if eye is not None:
            eye_views = self.eye_transform(eye)  # Multi-view embedding
            eye_cls = self.eye_cls_token.expand(batch_size, -1, -1)
            eye_tokens = torch.cat([eye_cls, eye_views], dim=1)
            features.append(eye_tokens)
        
        # Combine multimodal features
        if len(features) > 1:
            x = torch.cat(features, dim=1)  # Multimodal fusion
        else:
            x = features[0]  # Single modality
        
        # Transformer processing
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # Extract classification features (first token as class token)
        cls_features = x[:, 0, :]
        
        # Generate emotion predictions
        if len(features) > 1:
            # Multimodal: use shared classifier
            output = self.head(cls_features)
        else:
            # Unimodal: use modality-specific classifier
            if eeg is not None:
                output = self.head_eeg(cls_features)
            else:
                output = self.head_eye(cls_features)
        
        # Domain adversarial training for generalization
        if self.domain_generalization and self.training:
            reversed_features = gradient_reversal(cls_features, alpha_)
            domain_output = self.domain_classifier(reversed_features)
            return output, domain_output
        
        return output
```
