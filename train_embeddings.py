"""
Contrastive Learning Embedding Training Script
Train product embeddings using Skip-gram with Negative Sampling on session sequences.

Usage:
    python train_embeddings.py --epochs 10 --dim 64 --window 5
    
Output:
    embeddings/item_embeddings.npy - Product embedding matrix (N_products, dim)
    embeddings/product_ids.json - List of product IDs corresponding to embedding rows
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import json
import os
import argparse
from tqdm.auto import tqdm
import ast
import warnings
warnings.filterwarnings('ignore')

# ==============================
# CONFIGURATION
# ==============================
class Config:
    DATA_PATH = '/kaggle/input/rental-product-recommendation-system/'
    OUTPUT_DIR = 'embeddings'
    EMBEDDING_DIM = 64
    WINDOW_SIZE = 5
    NEGATIVE_SAMPLES = 5
    LEARNING_RATE = 0.025
    MIN_LR = 0.0001
    EPOCHS = 10
    MIN_COUNT = 2  # Minimum occurrences to include product
    SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(description='Train product embeddings')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--window', type=int, default=5, help='Context window size')
    parser.add_argument('--neg', type=int, default=5, help='Number of negative samples')
    parser.add_argument('--lr', type=float, default=0.025, help='Initial learning rate')
    parser.add_argument('--data-path', type=str, default=None, help='Override data path')
    parser.add_argument('--output-dir', type=str, default='embeddings', help='Output directory')
    return parser.parse_args()

# ==============================
# DATA LOADING
# ==============================
def load_data(data_path: str):
    """Load and prepare training data."""
    print("Loading data...")
    
    # Load behavioral data
    visits = pd.read_csv(f"{data_path}metrika_visits.csv")
    hits = pd.read_csv(f"{data_path}metrika_hits.csv", low_memory=False)
    
    # Load product catalogs
    new_products = pd.read_csv(f"{data_path}new_site_products.csv")
    old_products = pd.read_csv(f"{data_path}old_site_products.csv")
    product_mapping = pd.read_csv(f"{data_path}old_site_new_site_products.csv")
    
    print(f"  Visits: {len(visits):,}")
    print(f"  Hits: {len(hits):,}")
    print(f"  New products: {len(new_products):,}")
    
    return visits, hits, new_products, old_products, product_mapping

def build_mappings(new_products, old_products, product_mapping):
    """Build product ID mappings."""
    mappings = {
        'new_slug_to_id': dict(zip(new_products['slug'], new_products['id'])),
        'old_slug_to_id': dict(zip(old_products['slug'], old_products['id'])),
        'old_to_new_id': dict(zip(product_mapping['old_site_id'], product_mapping['new_site_id'])),
        'valid_new_ids': set(new_products['id'].tolist()),
    }
    
    # Old slug -> New ID mapping
    mappings['old_slug_to_new_id'] = {}
    for old_slug, old_id in mappings['old_slug_to_id'].items():
        if old_id in mappings['old_to_new_id']:
            mappings['old_slug_to_new_id'][old_slug] = mappings['old_to_new_id'][old_id]
    
    return mappings

def build_watch_to_visit(visits: pd.DataFrame) -> dict:
    """Build mapping from watch_id to visit_id."""
    watch_to_visit = {}
    for _, row in tqdm(visits.iterrows(), total=len(visits), desc="Building watch->visit"):
        visit_id = row['visit_id']
        try:
            watch_ids = ast.literal_eval(str(row['watch_ids']))
            for wid in watch_ids:
                watch_to_visit[int(wid)] = visit_id
        except:
            pass
    return watch_to_visit

# ==============================
# SESSION SEQUENCE EXTRACTION
# ==============================
def extract_sessions(hits: pd.DataFrame, visits: pd.DataFrame, mappings: dict) -> list:
    """Extract product sequences from sessions."""
    print("\nExtracting session sequences...")
    
    # Separate by site
    new_visits = visits[visits['project_id'] == 0]
    old_visits = visits[visits['project_id'] == 1]
    new_hits = hits[hits['project_id'] == 0]
    old_hits = hits[hits['project_id'] == 1]
    
    print(f"  New site: {len(new_visits):,} visits, {len(new_hits):,} hits")
    print(f"  Old site: {len(old_visits):,} visits, {len(old_hits):,} hits")
    
    # Build watch->visit mappings
    new_watch_to_visit = build_watch_to_visit(new_visits)
    old_watch_to_visit = build_watch_to_visit(old_visits)
    
    all_sequences = []
    
    # Process new site
    print("\nProcessing new site sessions...")
    new_hits = new_hits.copy()
    new_hits['visit_id'] = new_hits['watch_id'].map(new_watch_to_visit)
    new_hits['product_id'] = new_hits['slug'].map(mappings['new_slug_to_id'])
    new_hits['date_time'] = pd.to_datetime(new_hits['date_time'], format='ISO8601', errors='coerce')
    
    product_hits = new_hits[new_hits['page_type'] == 'PRODUCT'].copy()
    product_hits = product_hits.sort_values(['visit_id', 'date_time'])
    
    session_seqs = product_hits.groupby('visit_id')['product_id'].apply(
        lambda x: [int(p) for p in x.dropna().tolist()]
    )
    for seq in session_seqs:
        if len(seq) >= 2:
            all_sequences.append(seq)
    
    print(f"  New site sequences: {len(all_sequences):,}")
    
    # Process old site (map to new IDs)
    print("\nProcessing old site sessions (mapped to new IDs)...")
    old_hits = old_hits.copy()
    old_hits['visit_id'] = old_hits['watch_id'].map(old_watch_to_visit)
    old_hits['product_id'] = old_hits['slug'].map(mappings['old_slug_to_new_id'])
    old_hits['date_time'] = pd.to_datetime(old_hits['date_time'], format='ISO8601', errors='coerce')
    
    old_product_hits = old_hits[old_hits['page_type'] == 'PRODUCT'].copy()
    old_product_hits = old_product_hits.sort_values(['visit_id', 'date_time'])
    
    old_session_seqs = old_product_hits.groupby('visit_id')['product_id'].apply(
        lambda x: [int(p) for p in x.dropna().tolist()]
    )
    old_count = 0
    for seq in old_session_seqs:
        if len(seq) >= 2:
            # Filter to only include products that mapped successfully
            filtered_seq = [p for p in seq if p in mappings['valid_new_ids']]
            if len(filtered_seq) >= 2:
                all_sequences.append(filtered_seq)
                old_count += 1
    
    print(f"  Old site sequences (mapped): {old_count:,}")
    print(f"  Total sequences: {len(all_sequences):,}")
    
    return all_sequences

# ==============================
# VOCABULARY BUILDING
# ==============================
def build_vocabulary(sequences: list, min_count: int = 2) -> tuple:
    """Build vocabulary with frequency filtering."""
    print("\nBuilding vocabulary...")
    
    # Count product occurrences
    product_counts = Counter()
    for seq in sequences:
        product_counts.update(seq)
    
    print(f"  Total unique products: {len(product_counts):,}")
    
    # Filter by minimum count
    filtered_products = [p for p, c in product_counts.items() if c >= min_count]
    print(f"  Products with count >= {min_count}: {len(filtered_products):,}")
    
    # Create mappings
    product_to_idx = {p: i for i, p in enumerate(filtered_products)}
    idx_to_product = {i: p for p, i in product_to_idx.items()}
    
    # Create unigram distribution for negative sampling (^0.75 smoothing)
    total_count = sum(product_counts[p] for p in filtered_products)
    unigram_probs = np.array([
        (product_counts[p] ** 0.75) for p in filtered_products
    ])
    unigram_probs = unigram_probs / unigram_probs.sum()
    
    return product_to_idx, idx_to_product, unigram_probs

# ==============================
# SKIP-GRAM MODEL
# ==============================
class SkipGramModel:
    """Skip-gram with Negative Sampling for product embeddings."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        np.random.seed(seed)
        # Initialize embeddings with small random values
        self.W = (np.random.randn(vocab_size, embedding_dim) * 0.01).astype(np.float32)
        self.W_context = (np.random.randn(vocab_size, embedding_dim) * 0.01).astype(np.float32)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
    
    def sigmoid(self, x):
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
    
    def train_pair(self, target_idx: int, context_idx: int, neg_indices: np.ndarray, 
                   lr: float) -> float:
        """Train on one positive pair + negative samples."""
        # Get embeddings
        target_vec = self.W[target_idx]
        context_vec = self.W_context[context_idx]
        neg_vecs = self.W_context[neg_indices]
        
        # Positive sample
        pos_score = np.dot(target_vec, context_vec)
        pos_sigmoid = self.sigmoid(pos_score)
        pos_grad = (pos_sigmoid - 1) * context_vec
        context_grad = (pos_sigmoid - 1) * target_vec
        
        # Negative samples
        neg_scores = np.dot(neg_vecs, target_vec)
        neg_sigmoids = self.sigmoid(neg_scores)
        neg_grads = neg_sigmoids.reshape(-1, 1) * neg_vecs
        neg_grad_sum = neg_grads.sum(axis=0)
        
        # Update target embedding
        self.W[target_idx] -= lr * (pos_grad + neg_grad_sum)
        
        # Update context embeddings
        self.W_context[context_idx] -= lr * context_grad
        for i, neg_idx in enumerate(neg_indices):
            self.W_context[neg_idx] -= lr * neg_sigmoids[i] * target_vec
        
        # Compute loss for monitoring
        loss = -np.log(pos_sigmoid + 1e-10) - np.sum(np.log(1 - neg_sigmoids + 1e-10))
        return loss
    
    def get_embeddings(self) -> np.ndarray:
        """Return final embeddings (average of W and W_context)."""
        return (self.W + self.W_context) / 2

# ==============================
# TRAINING LOOP
# ==============================
def train_embeddings(sequences: list, product_to_idx: dict, unigram_probs: np.ndarray,
                    embedding_dim: int, window_size: int, neg_samples: int,
                    epochs: int, init_lr: float, min_lr: float, seed: int):
    """Train Skip-gram model."""
    
    vocab_size = len(product_to_idx)
    print(f"\nTraining Skip-gram model...")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Window size: {window_size}")
    print(f"  Negative samples: {neg_samples}")
    print(f"  Epochs: {epochs}")
    
    # Convert sequences to indices
    indexed_sequences = []
    for seq in sequences:
        indexed_seq = [product_to_idx[p] for p in seq if p in product_to_idx]
        if len(indexed_seq) >= 2:
            indexed_sequences.append(indexed_seq)
    
    # Count total training pairs
    total_pairs = sum(
        sum(min(i, window_size) + min(len(seq) - 1 - i, window_size) 
            for i in range(len(seq)))
        for seq in indexed_sequences
    )
    print(f"  Total training pairs: {total_pairs:,}")
    
    # Initialize model
    model = SkipGramModel(vocab_size, embedding_dim, seed)
    
    # Training
    np.random.seed(seed)
    processed_pairs = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_pairs = 0
        
        # Shuffle sequences
        np.random.shuffle(indexed_sequences)
        
        for seq in tqdm(indexed_sequences, desc=f"Epoch {epoch+1}/{epochs}"):
            for i, target_idx in enumerate(seq):
                # Dynamic window
                actual_window = np.random.randint(1, window_size + 1)
                
                # Context indices
                start = max(0, i - actual_window)
                end = min(len(seq), i + actual_window + 1)
                
                for j in range(start, end):
                    if i == j:
                        continue
                    
                    context_idx = seq[j]
                    
                    # Sample negatives
                    neg_indices = np.random.choice(
                        vocab_size, size=neg_samples, replace=False, p=unigram_probs
                    )
                    
                    # Learning rate decay
                    progress = processed_pairs / (total_pairs * epochs)
                    lr = max(min_lr, init_lr * (1 - progress))
                    
                    # Train
                    loss = model.train_pair(target_idx, context_idx, neg_indices, lr)
                    epoch_loss += loss
                    epoch_pairs += 1
                    processed_pairs += 1
        
        avg_loss = epoch_loss / max(epoch_pairs, 1)
        print(f"  Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")
    
    return model.get_embeddings()

# ==============================
# MAIN
# ==============================
def main():
    args = parse_args()
    
    # Determine data path
    if args.data_path:
        data_path = args.data_path
    elif os.path.exists('/kaggle/input/rental-product-recommendation-system/'):
        data_path = '/kaggle/input/rental-product-recommendation-system/'
    elif os.path.exists('data/'):
        data_path = 'data/'
    else:
        raise ValueError("Could not find data path. Use --data-path to specify.")
    
    print(f"Using data path: {data_path}")
    
    # Load data
    visits, hits, new_products, old_products, product_mapping = load_data(data_path)
    
    # Build mappings
    mappings = build_mappings(new_products, old_products, product_mapping)
    
    # Extract sessions
    sequences = extract_sessions(hits, visits, mappings)
    
    # Build vocabulary
    product_to_idx, idx_to_product, unigram_probs = build_vocabulary(
        sequences, min_count=Config.MIN_COUNT
    )
    
    # Train embeddings
    embeddings = train_embeddings(
        sequences=sequences,
        product_to_idx=product_to_idx,
        unigram_probs=unigram_probs,
        embedding_dim=args.dim,
        window_size=args.window,
        neg_samples=args.neg,
        epochs=args.epochs,
        init_lr=args.lr,
        min_lr=Config.MIN_LR,
        seed=Config.SEED
    )
    
    # Save embeddings
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    embeddings_path = os.path.join(output_dir, 'item_embeddings.npy')
    product_ids_path = os.path.join(output_dir, 'product_ids.json')
    
    # Save in order of idx_to_product
    product_ids = [idx_to_product[i] for i in range(len(idx_to_product))]
    
    np.save(embeddings_path, embeddings)
    with open(product_ids_path, 'w') as f:
        json.dump(product_ids, f)
    
    print(f"\n✓ Embeddings saved to {embeddings_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {os.path.getsize(embeddings_path) / 1024 / 1024:.2f} MB")
    print(f"✓ Product IDs saved to {product_ids_path}")
    print(f"  Products: {len(product_ids):,}")

if __name__ == '__main__':
    main()
