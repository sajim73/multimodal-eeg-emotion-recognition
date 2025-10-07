"""
Multi-view Embedding Module for MAET
Implements multi-view feature transformation with gating mechanism
"""

import torch
import torch.nn as nn


class MultiViewEmbedding(nn.Module):
    """
    Multi-view embedding module that creates multiple views of input features
    with adaptive gating mechanism for view selection and fusion.
    
    This module is based on the MAET paper's approach to create diverse
    representations of the same input features.
    
    Args:
        input_dim (int): Dimension of input features
        embed_dim (int): Output embedding dimension
        num_views (int): Number of different views to create (default: 5)
    """
    
    def __init__(self, input_dim, embed_dim, num_views=5):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_views = num_views
        
        # Create separate projection layers for each view
        self.view_projections = nn.ModuleList([
            nn.Linear(input_dim, embed_dim) for _ in range(num_views)
        ])
        
        # Gating mechanism to weight different views adaptively
        self.gate = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Batch normalization for stability
        self.batch_norm = nn.BatchNorm1d(num_views * embed_dim)
    
    def forward(self, x):
        """
        Forward pass through multi-view embedding
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Multi-view embeddings [batch_size, num_views, embed_dim]
        """
        batch_size = x.size(0)
        
        # Generate multiple views through different linear projections
        views = torch.stack([proj(x) for proj in self.view_projections], dim=1)
        # Shape: [batch_size, num_views, embed_dim]
        
        # Apply adaptive gating
        gate = self.gate(x).unsqueeze(1)  # [batch_size, 1, embed_dim]
        views = views * gate  # Element-wise multiplication with broadcasting
        
        # Apply batch normalization (flatten for BatchNorm1d)
        views_flat = views.reshape(batch_size, -1)
        views_flat = self.batch_norm(views_flat)
        
        # Reshape back to multi-view format
        return views_flat.reshape(batch_size, self.num_views, -1)
```
