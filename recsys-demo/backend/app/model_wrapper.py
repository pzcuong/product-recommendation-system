"""
CL-GRU4Rec+RP Model Wrapper for Real API
Simplified wrapper that uses popularity-based recommendations
when model is not available
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)


class ModelWrapper:
    """Wrapper for CL-GRU4Rec+RP model"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset: str = "kaggle_rental",
        device: str = "cpu"
    ):
        self.dataset = dataset
        self.device = torch.device(device)
        self.model = None
        self.item2idx = {}
        self.idx2item = {}
        self.num_items = 0
        self._initialized = False

        # Try to load the model
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]):
        """Load or initialize the model"""
        try:
            # Create item mappings from data
            data_path = Path("/Users/macbook/Desktop/product-recommendation-system/data")

            if self.dataset == "kaggle_rental":
                products_file = data_path / "new_site_products.csv"

                if products_file.exists():
                    import pandas as pd

                    # Load data for mappings
                    products_df = pd.read_csv(products_file)

                    # Create item mappings from product IDs
                    all_product_ids = products_df['id'].astype(str).unique().tolist()

                    self.item2idx = {pid: idx for idx, pid in enumerate(all_product_ids)}
                    self.idx2item = {idx: pid for pid, idx in self.item2idx.items()}
                    self.num_items = len(all_product_ids)

                    logger.info(f"Created mappings for {self.num_items} items")

            # Note: For demo purposes, we're using popularity-based recommendations
            # To use the actual trained model, you would need to:
            # 1. Train the model using the training scripts
            # 2. Save the checkpoint
            # 3. Load it here using torch.load()

            # Try to load checkpoint if provided
            if model_path and Path(model_path).exists():
                try:
                    from cl_gru4rec_rp_unified import GRU4RecModel

                    embedding_dim = 128
                    hidden_dim = 256

                    self.model = GRU4RecModel(
                        num_items=self.num_items,
                        embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim,
                        num_layers=2,
                        dropout=0.2
                    )

                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(checkpoint)
                    self.model.to(self.device)
                    self.model.eval()
                    self._initialized = True

                    logger.info(f"Loaded checkpoint from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}")
                    self._initialized = False
            else:
                # No checkpoint available - use popularity-based recommendations
                logger.info("No trained checkpoint available - using popularity-based recommendations")
                self._initialized = False

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self._initialized = False

    def predict(
        self,
        session_items: List[str],
        k: int = 10
    ) -> List[Dict]:
        """Generate recommendations for a session"""
        # For now, return empty list - the API will use popularity fallback
        return []

    def get_confidence(self, session_length: int) -> float:
        """Calculate prediction confidence based on session length"""
        # More items = higher confidence
        base_confidence = 0.3
        max_confidence = 0.95

        confidence = min(
            base_confidence + (session_length / 20) * 0.5,
            max_confidence
        )

        return confidence


# Global model instance
model_wrapper = None


def get_model() -> ModelWrapper:
    """Get or create global model instance"""
    global model_wrapper
    if model_wrapper is None:
        model_wrapper = ModelWrapper()
    return model_wrapper
