"""
Data Loader for CL-GRU4Rec+RP Real API
Loads real data from the Kaggle Rental Product dataset
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

DATA_PATH = Path("/Users/macbook/Desktop/product-recommendation-system/data")


class DataLoader:
    """Loads and manages product data from the real dataset"""

    def __init__(self):
        self.products_df: Optional[pd.DataFrame] = None
        self.hits_df: Optional[pd.DataFrame] = None
        self.product_map: Dict[str, Dict] = {}
        self._load_data()

    def _load_data(self):
        """Load products and session data from CSV files"""
        try:
            # Load products
            products_file = DATA_PATH / "new_site_products.csv"
            if products_file.exists():
                self.products_df = pd.read_csv(products_file)
                logger.info(f"Loaded {len(self.products_df)} products")

                # Build product map
                for _, row in self.products_df.iterrows():
                    self.product_map[str(row['id'])] = {
                        'id': str(row['id']),
                        'name': row.get('name', ''),
                        'brand': row.get('brand', ''),
                        'main_category': row.get('main_category', ''),
                        'categories': row.get('categories', ''),
                        'price': row.get('price_per_period_day', 0),
                        'price_sell': row.get('price_sell', 0),
                        'description': row.get('description', ''),
                        'slug': row.get('slug', ''),
                    }
            else:
                logger.warning(f"Products file not found: {products_file}")

            # Load hits (sessions) - sample for now
            hits_file = DATA_PATH / "metrika_hits.csv"
            if hits_file.exists():
                # Load only a sample for performance
                self.hits_df = pd.read_csv(hits_file, nrows=100000)
                logger.info(f"Loaded {len(self.hits_df)} session hits (sample)")

        except Exception as e:
            logger.error(f"Error loading data: {e}")

    def get_product(self, product_id: str) -> Optional[Dict]:
        """Get a single product by ID"""
        return self.product_map.get(product_id)

    def get_all_products(self) -> List[Dict]:
        """Get all products"""
        return list(self.product_map.values())

    def get_products_by_category(self, category: str) -> List[Dict]:
        """Get products filtered by main category"""
        return [
            p for p in self.product_map.values()
            if p.get('main_category') == category
        ]

    def get_categories(self) -> List[str]:
        """Get all unique main categories"""
        if self.products_df is None:
            return []
        return self.products_df['main_category'].dropna().unique().tolist()

    def get_session_history(self, client_id: str, limit: int = 50) -> List[str]:
        """Get session history for a client"""
        if self.hits_df is None:
            return []

        try:
            client_hits = self.hits_df[
                self.hits_df['client_id'].astype(str) == str(client_id)
            ].head(limit)

            # Extract product IDs from URLs
            product_ids = []
            for _, row in client_hits.iterrows():
                url = row.get('url', '')
                if '/product/' in url or url.endswith('.html'):
                    # Try to extract product ID from URL
                    parts = url.split('/')[-1].replace('.html', '').split('-')
                    if parts:
                        product_ids.append(parts[-1])

            return product_ids
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []

    def get_popular_products(self, limit: int = 50) -> List[str]:
        """Get most popular product IDs"""
        if self.hits_df is None:
            return list(self.product_map.keys())[:limit]

        try:
            # Count product views
            product_views = {}

            for _, row in self.hits_df.head(50000).iterrows():
                url = row.get('url', '')
                page_type = row.get('page_type', '')

                if page_type == 'PRODUCT':
                    # Extract product ID
                    parts = url.split('/')[-1].replace('.html', '').split('-')
                    if parts:
                        pid = parts[-1]
                        product_views[pid] = product_views.get(pid, 0) + 1

            # Sort by view count
            sorted_products = sorted(
                product_views.items(),
                key=lambda x: x[1],
                reverse=True
            )

            return [pid for pid, _ in sorted_products[:limit]]
        except Exception as e:
            logger.error(f"Error getting popular products: {e}")
            return list(self.product_map.keys())[:limit]


# Global data loader instance
data_loader = DataLoader()
