"""UniSRec: Universal Sequence Representation for Recommendation.

Transfer learning-enabled SASRec variant using BERT embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear
        output = self.out_linear(output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class UniSRec(nn.Module):
    """UniSRec model for universal sequence recommendation."""
    
    def __init__(
        self,
        embedding_dim: int = 768,  # BERT dimension
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        max_seq_length: int = 50,
        use_moe: bool = True,
        num_experts: int = 4
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Projection from BERT embeddings to hidden dimension
        if use_moe:
            from src.text_encoder import MoEProjection
            self.projection = MoEProjection(embedding_dim, hidden_dim, num_experts)
        else:
            from src.text_encoder import LearnableProjection
            self.projection = LearnableProjection(embedding_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_length)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                d_ff=hidden_dim * 4,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For prediction
        self.output_bias = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        item_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            item_embeddings: Tensor of shape (batch, seq_len, embedding_dim)
                            Pre-computed BERT embeddings
            attention_mask: Optional mask of shape (batch, seq_len)
            
        Returns:
            Sequence representations of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = item_embeddings.size()
        
        # Project to hidden dimension
        x = self.projection(item_embeddings)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create causal attention mask
        causal_mask = self._create_causal_mask(seq_len, x.device)
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Broadcast to (batch, num_heads, seq_len, seq_len)
            padding_mask = padding_mask.expand(-1, 1, seq_len, -1)
            # Combine masks
            mask = causal_mask.unsqueeze(0) * padding_mask
        else:
            mask = causal_mask.unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output normalization
        x = self.output_norm(x)
        x = self.dropout(x)
        
        return x
    
    def predict(
        self,
        sequence_output: torch.Tensor,
        candidate_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict scores for candidate items.
        
        Args:
            sequence_output: Output from forward pass (batch, seq_len, hidden_dim)
            candidate_embeddings: Candidate item embeddings (batch, num_candidates, embedding_dim)
            
        Returns:
            Scores of shape (batch, seq_len, num_candidates)
        """
        # Project candidates
        candidate_hidden = self.projection(candidate_embeddings)  # (batch, num_candidates, hidden_dim)
        
        # Dot product: (batch, seq_len, hidden_dim) @ (batch, hidden_dim, num_candidates)
        scores = torch.matmul(sequence_output, candidate_hidden.transpose(1, 2))
        
        # Add bias
        scores = scores + self.output_bias
        
        return scores
    
    def _create_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Create causal (lower-triangular) attention mask.
        
        Args:
            size: Sequence length
            device: torch device
            
        Returns:
            Mask tensor of shape (size, size)
        """
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask


class SampledSoftmaxLoss(nn.Module):
    """Sampled softmax loss for training."""
    
    def __init__(self, num_negatives: int = 2048):
        super().__init__()
        self.num_negatives = num_negatives
    
    def forward(
        self,
        sequence_output: torch.Tensor,
        target_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
        projection_layer: nn.Module
    ) -> torch.Tensor:
        """
        Compute sampled softmax loss.
        
        Args:
            sequence_output: Sequence representations (batch, seq_len, hidden_dim)
            target_embeddings: Target item embeddings (batch, embedding_dim)
            negative_embeddings: Negative item embeddings (batch, num_negatives, embedding_dim)
            projection_layer: Projection layer to apply to embeddings
            
        Returns:
            Loss scalar
        """
        batch_size = sequence_output.size(0)
        
        # Use last position for prediction
        last_output = sequence_output[:, -1, :]  # (batch, hidden_dim)
        
        # Project target and negatives
        target_hidden = projection_layer(target_embeddings)  # (batch, hidden_dim)
        negative_hidden = projection_layer(negative_embeddings)  # (batch, num_negatives, hidden_dim)
        
        # Compute scores
        pos_score = (last_output * target_hidden).sum(dim=1)  # (batch,)
        neg_scores = torch.matmul(negative_hidden, last_output.unsqueeze(-1)).squeeze(-1)  # (batch, num_negatives)
        
        # Sampled softmax loss
        logits = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)  # (batch, 1 + num_negatives)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # Target is always index 0
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


if __name__ == "__main__":
    # Test model
    batch_size = 4
    seq_len = 10
    embedding_dim = 768
    hidden_dim = 256
    
    model = UniSRec(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_blocks=2,
        num_heads=4,
        dropout=0.3
    )
    
    # Dummy input
    item_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Forward pass
    output = model(item_embeddings)
    
    print(f"Input shape: {item_embeddings.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    num_candidates = 20
    candidate_embeddings = torch.randn(batch_size, num_candidates, embedding_dim)
    scores = model.predict(output, candidate_embeddings)
    
    print(f"Prediction scores shape: {scores.shape}")
