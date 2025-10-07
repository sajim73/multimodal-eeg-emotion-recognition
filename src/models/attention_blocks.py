"""
Transformer Attention Blocks for MAET
Implements self-attention mechanisms for multimodal feature processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    Simplified transformer block with multi-head self-attention and feed-forward network.
    
    This module processes sequences of embedded features through attention mechanisms
    to capture long-range dependencies and contextual relationships.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through transformer block
        
        Args:
            x (torch.Tensor): Input features [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: Transformed features [batch_size, seq_len, embed_dim]
        """
        # Multi-head self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward network with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer inputs to provide sequence position information.
    
    Args:
        embed_dim (int): Embedding dimension
        max_len (int): Maximum sequence length (default: 512)
    """
    
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x (torch.Tensor): Input embeddings [seq_len, batch_size, embed_dim]
            
        Returns:
            torch.Tensor: Position-encoded embeddings
        """
        x = x + self.pe[:x.size(0), :]
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing EEG and eye tracking features.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability (default: 0.1)
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Cross-attention layers
        self.eeg_to_eye_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.eye_to_eeg_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, eeg_features, eye_features):
        """
        Perform cross-modal attention between EEG and eye tracking features
        
        Args:
            eeg_features (torch.Tensor): EEG features [batch_size, seq_len, embed_dim]
            eye_features (torch.Tensor): Eye features [batch_size, seq_len, embed_dim]
            
        Returns:
            tuple: Enhanced (eeg_features, eye_features)
        """
        # EEG queries attend to eye tracking keys/values
        eeg_enhanced, _ = self.eeg_to_eye_attn(
            eeg_features, eye_features, eye_features
        )
        eeg_features = self.norm1(eeg_features + self.dropout(eeg_enhanced))
        
        # Eye tracking queries attend to EEG keys/values
        eye_enhanced, _ = self.eye_to_eeg_attn(
            eye_features, eeg_features, eeg_features
        )
        eye_features = self.norm2(eye_features + self.dropout(eye_enhanced))
        
        return eeg_features, eye_features
```
