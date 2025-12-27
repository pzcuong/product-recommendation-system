"""Data processing pipeline for recommendation system."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.utils import parse_datetime, get_season, compute_age_delta, normalize_embeddings


class DataProcessor:
    """Main data processing pipeline."""
    
    def __init__(self, config: dict):
        """
        Initialize data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_path = config['data']['data_path']
        self.recency_cutoff = config['data']['recency_cutoff']
        
        # Data containers
        self.products = None
        self.sessions = None
        self.product_embeddings = None
        self.product_features = {}
        
        print(f"DataProcessor initialized with recency cutoff: {self.recency_cutoff}")
    
    def load_all_data(self) -> Dict:
        """
        Load all data files.
        
        Returns:
            Dictionary with all loaded dataframes
        """
        print("\n=== Loading Data ===")
        
        data = {}
        
        # Products
        print("Loading products...")
        data['new_products'] = pd.read_csv(f"{self.data_path}new_site_products.csv")
        data['old_products'] = pd.read_csv(f"{self.data_path}old_site_products.csv")
        
        # Product mapping
        data['product_mapping'] = pd.read_csv(f"{self.data_path}old_site_new_site_products.csv")
        
        # Orders
        print("Loading orders...")
        data['new_orders'] = pd.read_csv(f"{self.data_path}new_site_orders.csv")
        data['old_orders'] = pd.read_csv(f"{self.data_path}old_site_orders.csv")
        data['old_carts'] = pd.read_csv(f"{self.data_path}old_site_carts.csv")
        
        # Sessions
        print("Loading sessions...")
        data['visits'] = pd.read_csv(f"{self.data_path}metrika_visits.csv")
        data['hits'] = pd.read_csv(f"{self.data_path}metrika_hits.csv")
        
        # Test data
        print("Loading test data...")
        data['test_visits'] = pd.read_csv(f"{self.data_path}metrika_visits_test.csv")
        data['test_hits'] = pd.read_csv(f"{self.data_path}metrika_hits_test.csv")
        
        print(f"✓ Loaded {len(data['new_products'])} new products")
        print(f"✓ Loaded {len(data['old_products'])} old products")
        print(f"✓ Loaded {len(data['visits'])} train visits")
        print(f"✓ Loaded {len(data['test_visits'])} test visits")
        
        return data
    
    def build_product_mappings(self, data: Dict) -> Dict:
        """
        Build product ID mappings between old and new sites.
        
        Args:
            data: Dictionary with loaded dataframes
            
        Returns:
            Dictionary with mappings
        """
        print("\n=== Building Product Mappings ===")
        
        mappings = {}
        
        # Old slug -> ID
        mappings['old_slug_to_id'] = dict(
            zip(data['old_products']['slug'], data['old_products']['id'])
        )
        
        # New slug -> ID
        mappings['new_slug_to_id'] = dict(
            zip(data['new_products']['slug'], data['new_products']['id'])
        )
        
        # Old ID -> New ID
        mappings['old_to_new_id'] = dict(
            zip(data['product_mapping']['old_site_id'],
                data['product_mapping']['new_site_id'])
        )
        
        # Old slug -> New ID (combined)
        mappings['old_slug_to_new_id'] = {}
        for old_slug, old_id in mappings['old_slug_to_id'].items():
            if old_id in mappings['old_to_new_id']:
                mappings['old_slug_to_new_id'][old_slug] = mappings['old_to_new_id'][old_id]
        
        # Valid new IDs
        mappings['valid_new_ids'] = set(data['new_products']['id'].tolist())
        
        print(f"✓ Built mappings: {len(mappings['old_to_new_id'])} old->new products")
        
        return mappings
    
    def extract_rental_features(self, data: Dict) -> Dict:
        """
        Extract rental-specific features from products.
        
        Args:
            data: Dictionary with loaded dataframes
            
        Returns:
            Dictionary with product features
        """
        print("\n=== Extracting Rental Features ===")
        
        features = {}
        products_df = data['new_products']
        
        for _, row in tqdm(products_df.iterrows(), total=len(products_df),
                          desc="Processing products"):
            pid = row['id']
            
            features[pid] = {
                'product_id': pid,
                'name': row.get('name', ''),
                'main_category': row.get('main_category', ''),
                'brand': row.get('brand', ''),
                'description': row.get('description', ''),
                
                # Age features (these would need to be extracted from metadata)
                # For now, using placeholder values - would be extracted from product descriptions
                'age_min_months': None,
                'age_max_months': None,
                
                # Seasonal features (example: outdoor products for summer)
                'is_seasonal': self._detect_seasonal(row),
                'season': self._get_product_season(row),
                
                # Rental duration hints (would be learned from historical data)
                'typical_rental_days': None,
            }
        
        print(f"✓ Extracted features for {len(features)} products")
        
        return features
    
    def _detect_seasonal(self, product_row) -> bool:
        """Detect if product is seasonal based on name/category."""
        text = str(product_row.get('name', '')) + ' ' + str(product_row.get('main_category', ''))
        text = text.lower()
        
        seasonal_keywords = ['зимний', 'летний', 'пляж', 'beach', 'winter', 'summer']
        return any(keyword in text for keyword in seasonal_keywords)
    
    def _get_product_season(self, product_row) -> Optional[str]:
        """Get primary season for product."""
        text = str(product_row.get('name', '')) + ' ' + str(product_row.get('main_category', ''))
        text = text.lower()
        
        if any(kw in text for kw in ['зимний', 'winter']):
            return 'winter'
        elif any(kw in text for kw in ['летний', 'summer', 'пляж', 'beach']):
            return 'summer'
        return None
    
    def build_sessions(self, data: Dict, mappings: Dict, is_test: bool = False) -> List[Dict]:
        """
        Build session sequences from hits data.
        
        Args:
            data: Dictionary with loaded dataframes
            mappings: Product mappings
            is_test: Whether processing test data
            
        Returns:
            List of session dictionaries
        """
        print(f"\n=== Building {'Test' if is_test else 'Train'} Sessions ===")
        
        # Select appropriate data
        if is_test:
            visits_df = data['test_visits']
            hits_df = data['test_hits']
        else:
            visits_df = data['visits']
            hits_df = data['hits']
        
        # Build watch_id -> visit_id mapping
        print("Building watch-to-visit mapping...")
        import json
        watch_to_visit = {}
        for _, row in visits_df.iterrows():
            try:
                # Handle both string JSON and potential lists/floats
                wids_raw = row['watch_ids']
                if pd.isna(wids_raw): continue
                
                if isinstance(wids_raw, str) and wids_raw.startswith('['):
                    wids = json.loads(wids_raw)
                else:
                    wids = [wids_raw]
                    
                for wid in wids:
                    try:
                        watch_to_visit[int(float(wid))] = row['visit_id']
                    except: pass
            except: pass
        
        # Process hits
        print("Processing hits...")
        hits_df = hits_df.copy()
        hits_df['visit_id'] = hits_df['watch_id'].map(watch_to_visit)
        
        # Map product slugs to IDs
        hits_df['product_id'] = None
        
        # Try new site mapping first
        mask_new = hits_df['url'].str.contains('/new/')
        for idx in tqdm(hits_df[mask_new].index, desc="Mapping new site products"):
            url = hits_df.loc[idx, 'url']
            slug = self._extract_slug(url)
            if slug in mappings['new_slug_to_id']:
                hits_df.loc[idx, 'product_id'] = mappings['new_slug_to_id'][slug]
        
        # Try old site mapping
        mask_old = ~mask_new
        for idx in tqdm(hits_df[mask_old].index, desc="Mapping old site products"):
            url = hits_df.loc[idx, 'url']
            slug = self._extract_slug(url)
            if slug in mappings['old_slug_to_new_id']:
                hits_df.loc[idx, 'product_id'] = mappings['old_slug_to_new_id'][slug]
        
        # Filter out unmapped products
        hits_df = hits_df[hits_df['product_id'].notna()]
        hits_df['product_id'] = hits_df['product_id'].astype(int)
        
        # Apply recency filter for training data
        if not is_test and self.recency_cutoff:
            print(f"Applying recency filter: {self.recency_cutoff}")
            if 'visit_start_time' in visits_df.columns:
                visits_df['visit_date'] = pd.to_datetime(visits_df['visit_start_time'])
                recent_visits = visits_df[visits_df['visit_date'] >= self.recency_cutoff]['visit_id']
                hits_df = hits_df[hits_df['visit_id'].isin(recent_visits)]
                print(f"  Retained {len(hits_df)} hits after recency filter")
        
        # Group by session
        print("Grouping by session...")
        sessions = []
        
        for visit_id, group in tqdm(hits_df.groupby('visit_id'), desc="Building sessions"):
            # Sort by timestamp if available
            if 'event_time' in group.columns:
                group = group.sort_values('event_time')
            
            product_sequence = group['product_id'].tolist()
            
            # Remove duplicates while preserving order
            seen = set()
            unique_sequence = []
            for pid in product_sequence:
                if pid not in seen:
                    unique_sequence.append(pid)
                    seen.add(pid)
            
            # Skip very short sessions
            if len(unique_sequence) < self.config['data']['min_session_length']:
                continue
            
            # Truncate long sessions
            if len(unique_sequence) > self.config['data']['max_session_length']:
                unique_sequence = unique_sequence[:self.config['data']['max_session_length']]
            
            sessions.append({
                'visit_id': visit_id,
                'products': unique_sequence,
                'length': len(unique_sequence),
            })
        
        print(f"✓ Built {len(sessions)} sessions")
        print(f"  Mean length: {np.mean([s['length'] for s in sessions]):.1f}")
        print(f"  Median length: {np.median([s['length'] for s in sessions]):.1f}")
        
        return sessions
    
    def _extract_slug(self, url: str) -> str:
        """Extract product slug from URL."""
        if pd.isna(url):
            return ""
        
        # Remove query parameters
        url = url.split('?')[0]
        
        # Get last part of path
        parts = url.rstrip('/').split('/')
        if parts:
            return parts[-1]
        return ""
    
    def create_train_val_split(
        self,
        sessions: List[Dict],
        val_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Create time-based train/validation split.
        
        Args:
            sessions: List of session dictionaries
            val_ratio: Ratio for validation set
            
        Returns:
            Tuple of (train_sessions, val_sessions)
        """
        print(f"\n=== Creating Train/Val Split ===")
        
        # Sort by visit_id (proxy for time)
        sessions_sorted = sorted(sessions, key=lambda x: x['visit_id'])
        
        # Split
        split_idx = int(len(sessions_sorted) * (1 - val_ratio))
        train_sessions = sessions_sorted[:split_idx]
        val_sessions = sessions_sorted[split_idx:]
        
        print(f"✓ Train: {len(train_sessions)} sessions")
        print(f"✓ Val: {len(val_sessions)} sessions")
        
        return train_sessions, val_sessions
    
    def build_cooccurrence_matrix(self, sessions: List[Dict]) -> Dict[int, Counter]:
        """
        Build co-occurrence matrix from sessions.
        
        Args:
            sessions: List of session dictionaries
            
        Returns:
            Dictionary mapping product -> Counter of co-occurring products
        """
        print("\n=== Building Co-occurrence Matrix ===")
        
        cooccur = defaultdict(Counter)
        
        for session in tqdm(sessions, desc="Processing sessions"):
            products = session['products']
            
            # All pairs in session
            for i, p1 in enumerate(products):
                for p2 in products[i+1:]:
                    cooccur[p1][p2] += 1
                    cooccur[p2][p1] += 1
        
        print(f"✓ Built co-occurrence for {len(cooccur)} products")
        
        return dict(cooccur)
    
    def build_transition_matrix(self, sessions: List[Dict]) -> Dict[int, Counter]:
        """
        Build sequential transition matrix from sessions.
        
        Args:
            sessions: List of session dictionaries
            
        Returns:
            Dictionary mapping product -> Counter of next products
        """
        print("\n=== Building Transition Matrix ===")
        
        transitions = defaultdict(Counter)
        
        for session in tqdm(sessions, desc="Processing sessions"):
            products = session['products']
            
            # Sequential transitions
            for i in range(len(products) - 1):
                transitions[products[i]][products[i+1]] += 1
        
        print(f"✓ Built transitions for {len(transitions)} products")
        
        return dict(transitions)


if __name__ == "__main__":
    # Test data processor
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    processor = DataProcessor(config)
    
    # Load data
    data = processor.load_all_data()
    
    # Build mappings
    mappings = processor.build_product_mappings(data)
    
    # Build sessions
    sessions = processor.build_sessions(data, mappings, is_test=False)
    
    print(f"\nProcessed {len(sessions)} sessions")
