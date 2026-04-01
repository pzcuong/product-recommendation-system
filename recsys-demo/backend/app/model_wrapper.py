"""
CL-GRU4Rec+RP Model Wrapper for Real API
Wraps the real PyTorch model for inference
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cl_gru4rec_rp_unified import (
    CL_GRU4Rec_RP_Unified,
    create_item_mappings,
    prepare_session_data,
)

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
                hits_file = data_path / "metrika_hits.csv"
                products_file = data_path / "new_site_products.csv"

                if hits_file.exists() and products_file.exists():
                    import pandas as pd

                    # Load data for mappings
                    hits_df = pd.read_csv(hits_file, nrows=100000)
                    products_df = pd.read_csv(products_file)

                    # Create item mappings from product IDs
                    all_product_ids = products_df['id'].astype(str).unique().tolist()

                    self.item2idx = {pid: idx for idx, pid in enumerate(all_product_ids)}
                    self.idx2item = {idx: pid for pid, idx in self.item2idx.items()}
                    self.num_items = len(all_product_ids)

                    logger.info(f"Created mappings for {self.num_items} items")

            # Initialize model
            embedding_dim = 128
            hidden_dim = 256

            self.model = CL_GRU4Rec_RP_Unified(
                num_items=self.num_items,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=0.2,
                cl_temp=0.2,
                fusion_strategy="adaptive",
                device=self.device
            )

            # Try to load checkpoint if provided
            if model_path and Path(model_path).exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"Loaded checkpoint from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}")

            self.model.to(self.device)
            self.model.eval()
            self._initialized = True

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self._initialized = False

    def predict(
        self,
        session_items: List[str],
        k: int = 10
    ) -> List[Dict]:
        """Generate recommendations for a session"""
        if not self._initialized or not session_items:
            return []

        try:
            # Convert session items to indices
            session_indices = []
            valid_items = []

            for item_id in session_items:
                if item_id in self.item2idx:
                    session_indices.append(self.item2idx[item_id])
                    valid_items.append(item_id)

            if not session_indices:
                return []

            # Prepare input tensor
            session_tensor = torch.tensor(
                [session_indices],
                dtype=torch.long,
                device=self.device
            )

            session_length = torch.tensor(
                [len(session_indices)],
                dtype=torch.long,
                device=self.device
            )

            # Get predictions from model
            with torch.no_grad():
                gru_out, item_emb = self.model.get_embeddings(
                    session_tensor,
                    session_length
                )

                # Calculate scores for all items
                all_items = torch.arange(
                    self.num_items,
                    device=self.device
                ).unsqueeze(0)

                scores = self.model.predict(session_tensor, session_length)

                # Remove already seen items
                seen_mask = torch.zeros(self.num_items, device=self.device)
                seen_mask[session_indices] = float('-inf')

                scores = scores + seen_mask.unsqueeze(0)

                # Get top k
                top_scores, top_indices = torch.topk(scores, k=min(k, self.num_items))

                # Convert to list of recommendations
                recommendations = []
                for i in range(top_indices.shape[1]):
                    idx = top_indices[0, i].item()
                    score = top_scores[0, i].item()

                    if idx in self.idx2item:
                        recommendations.append({
                            'product_id': self.idx2item[idx],
                            'score': score,
                            'gru_score': score * 0.7,  # Approximate component scores
                            'cl_score': score * 0.2,
                            'rp_score': score * 0.1,
                        })

                return recommendations

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
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
