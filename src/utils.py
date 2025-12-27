"""Utility functions for the recommendation system."""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_pickle(obj, path: str):
    """Save object to pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """Load object from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def calculate_recall_at_k(predictions: List[List], ground_truth: List, k: int = 6) -> float:
    """
    Calculate Recall@K.
    
    Args:
        predictions: List of prediction lists for each session
        ground_truth: List of ground truth items
        k: Top-K to consider
        
    Returns:
        Recall@K score
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    hits = 0
    for pred, gt in zip(predictions, ground_truth):
        if gt in pred[:k]:
            hits += 1
    
    return hits / len(predictions) if len(predictions) > 0 else 0.0


def calculate_ndcg_at_k(predictions: List[List], ground_truth: List, k: int = 6) -> float:
    """
    Calculate NDCG@K.
    
    Args:
        predictions: List of prediction lists for each session
        ground_truth: List of ground truth items
        k: Top-K to consider
        
    Returns:
        NDCG@K score
    """
    ndcg_sum = 0.0
    
    for pred, gt in zip(predictions, ground_truth):
        # DCG
        dcg = 0.0
        for i, item in enumerate(pred[:k]):
            if item == gt:
                dcg = 1.0 / np.log2(i + 2)  # i+2 because index starts at 0
                break
        
        # IDCG (ideal is 1.0 at position 0)
        idcg = 1.0
        
        ndcg_sum += dcg / idcg
    
    return ndcg_sum / len(predictions) if len(predictions) > 0 else 0.0


def calculate_coverage(predictions: List[List], total_items: int) -> float:
    """
    Calculate catalog coverage.
    
    Args:
        predictions: List of prediction lists
        total_items: Total number of items in catalog
        
    Returns:
        Coverage ratio (0-1)
    """
    unique_items = set()
    for pred in predictions:
        unique_items.update(pred)
    
    return len(unique_items) / total_items if total_items > 0 else 0.0


def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string with flexible format handling."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Unable to parse datetime: {dt_str}")


def create_submission_df(visit_ids: List[int], predictions: List[List[int]]) -> pd.DataFrame:
    """
    Create submission DataFrame in the required format.
    
    Args:
        visit_ids: List of visit IDs
        predictions: List of prediction lists (each with 6 items)
        
    Returns:
        DataFrame with columns: visit_id, product_ids
    """
    if len(visit_ids) != len(predictions):
        raise ValueError("visit_ids and predictions must have same length")
    
    # Format predictions as space-separated strings
    product_ids_str = []
    for pred in predictions:
        # Ensure exactly 6 predictions
        pred_truncated = pred[:6]
        # Pad with -1 if less than 6 (shouldn't happen but safety check)
        while len(pred_truncated) < 6:
            pred_truncated.append(-1)
        
        product_ids_str.append(' '.join(map(str, pred_truncated)))
    
    df = pd.DataFrame({
        'visit_id': visit_ids,
        'product_ids': product_ids_str
    })
    
    return df


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit sphere.
    
    Args:
        embeddings: Array of shape (N, D)
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
    return embeddings / norms


def get_season(month: int) -> str:
    """
    Get season from month (Northern Hemisphere).
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season name: 'winter', 'spring', 'summer', 'fall'
    """
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'


def compute_age_delta(age_min_1: Optional[int], age_max_1: Optional[int],
                      age_min_2: Optional[int], age_max_2: Optional[int]) -> float:
    """
    Compute age progression delta between two products.
    
    Args:
        age_min_1, age_max_1: Age range for product 1
        age_min_2, age_max_2: Age range for product 2
        
    Returns:
        Age delta (positive means progression forward)
    """
    if any(x is None for x in [age_min_1, age_max_1, age_min_2, age_max_2]):
        return 0.0
    
    # Use mid-point of age ranges
    mid_1 = (age_min_1 + age_max_1) / 2
    mid_2 = (age_min_2 + age_max_2) / 2
    
    return mid_2 - mid_1


class EarlyStopping:
    """Early stopping helper."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like recall, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current score
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
