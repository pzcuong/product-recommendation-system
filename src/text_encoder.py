"""Text encoding using BERT for product embeddings."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


class BERTEncoder:
    """BERT-based text encoder for product representations."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        device: str = "cpu",
        max_length: int = 128,
        batch_size: int = 32
    ):
        """
        Initialize BERT encoder.
        
        Args:
            model_name: HuggingFace model name
            device: torch device
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
        """
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        print(f"Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Freeze weights for efficiency
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.embedding_dim = self.model.config.hidden_size
        print(f"BERT embedding dim: {self.embedding_dim}")
    
    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode list of texts to embeddings.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        embeddings = []
        
        # Process in batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(texts), self.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts", total=num_batches)
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get embeddings (use [CLS] token)
                outputs = self.model(**encoded)
                # outputs.last_hidden_state shape: (batch, seq_len, hidden_size)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
                
                embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def create_product_text(
        self,
        title: str,
        category: Optional[str] = None,
        description: Optional[str] = None,
        brand: Optional[str] = None
    ) -> str:
        """
        Create concatenated text representation for a product.
        
        Args:
            title: Product title
            category: Product category
            description: Product description
            brand: Product brand
            
        Returns:
            Concatenated text string
        """
        parts = []
        
        if title and pd.notna(title):
            parts.append(str(title))
        
        if category and pd.notna(category):
            parts.append(str(category))
        
        if description and pd.notna(description):
            # Truncate long descriptions
            desc = str(description)
            if len(desc) > 200:
                desc = desc[:200]
            parts.append(desc)
        
        if brand and pd.notna(brand):
            parts.append(str(brand))
        
        return " ".join(parts) if parts else "unknown product"


def generate_product_embeddings(
    products_df: pd.DataFrame,
    encoder: BERTEncoder,
    cache_path: Optional[str] = None
) -> Dict[int, np.ndarray]:
    """
    Generate BERT embeddings for all products.
    
    Args:
        products_df: DataFrame with product metadata
        encoder: BERTEncoder instance
        cache_path: Path to cache embeddings (parquet)
        
    Returns:
        Dictionary mapping product_id -> embedding array
    """
    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"Loading embeddings from cache: {cache_path}")
        cached_df = pd.read_parquet(cache_path)
        embeddings_dict = {}
        for _, row in cached_df.iterrows():
            embeddings_dict[row['product_id']] = np.array(row['embedding'])
        print(f"Loaded {len(embeddings_dict)} cached embeddings")
        return embeddings_dict
    
    print(f"Generating embeddings for {len(products_df)} products...")
    
    # Create text representations
    texts = []
    product_ids = []
    
    for _, row in products_df.iterrows():
        text = encoder.create_product_text(
            title=row.get('name', ''),
            category=row.get('main_category', ''),
            description=row.get('description', ''),
            brand=row.get('brand', '')
        )
        texts.append(text)
        product_ids.append(row['id'])
    
    # Encode
    embeddings = encoder.encode_texts(texts, show_progress=True)
    
    # Create dictionary
    embeddings_dict = {
        pid: emb for pid, emb in zip(product_ids, embeddings)
    }
    
    # Cache if path provided
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache_df = pd.DataFrame({
            'product_id': product_ids,
            'embedding': [emb.tolist() for emb in embeddings]
        })
        cache_df.to_parquet(cache_path, index=False)
        print(f"Cached embeddings to: {cache_path}")
    
    return embeddings_dict


class LearnableProjection(torch.nn.Module):
    """Learnable projection layer for BERT embeddings."""
    
    def __init__(self, input_dim: int = 768, output_dim: int = 256):
        """
        Args:
            input_dim: Input embedding dimension (BERT hidden size)
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        # Simple linear projection
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.LayerNorm(output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Projected tensor of shape (batch, output_dim)
        """
        return self.projection(x)


class MoEProjection(torch.nn.Module):
    """Mixture-of-Experts projection for domain adaptation."""
    
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 256,
        num_experts: int = 4
    ):
        """
        Args:
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension
            num_experts: Number of expert networks
        """
        super().__init__()
        
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(input_dim, num_experts),
            torch.nn.Softmax(dim=-1)
        )
        
        # Layer norm
        self.norm = torch.nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project with mixture of experts.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Projected tensor of shape (batch, output_dim)
        """
        # Get expert weights
        gate_weights = self.gate(x)  # (batch, num_experts)
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # Each: (batch, output_dim)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, output_dim)
        
        # Weighted combination
        gate_weights = gate_weights.unsqueeze(-1)  # (batch, num_experts, 1)
        output = (expert_outputs * gate_weights).sum(dim=1)  # (batch, output_dim)
        
        return self.norm(output)


if __name__ == "__main__":
    # Test encoding
    encoder = BERTEncoder()
    
    test_texts = [
        "Коляска YoYo для новорожденных",
        "Детская кроватка Snoo Bassinet",
        "Автокресло группа 0+"
    ]
    
    embeddings = encoder.encode_texts(test_texts, show_progress=False)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dim: {embeddings.shape[1]}")
