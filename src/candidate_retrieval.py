"""Candidate retrieval using hybrid methods."""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import faiss
from tqdm.auto import tqdm

from src.utils import normalize_embeddings


class CandidateRetriever:
    """Hybrid candidate retrieval system."""
    
    def __init__(self, config: dict):
        """
        Initialize retriever.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.num_candidates = config['retrieval']['num_candidates']
        self.faiss_k = config['retrieval']['faiss_k']
        
        # Storage
        self.product_embeddings = None
        self.product_ids = None
        self.faiss_index = None
        self.cooccur_matrix = None
        self.transition_matrix = None
        
        print(f"CandidateRetriever initialized (k={self.num_candidates})")
    
    def build_faiss_index(self, product_embeddings: Dict[int, np.ndarray]):
        """
        Build FAISS index for semantic search.
        
        Args:
            product_embeddings: Dictionary mapping product_id -> embedding
        """
        print("\n=== Building FAISS Index ===")
        
        # Convert to arrays
        self.product_ids = sorted(product_embeddings.keys())
        embeddings_list = [product_embeddings[pid] for pid in self.product_ids]
        embeddings_array = np.vstack(embeddings_list).astype('float32')
        
        # Normalize for inner product search
        embeddings_array = normalize_embeddings(embeddings_array)
        
        # Build index
        embedding_dim = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product
        self.faiss_index.add(embeddings_array)
        
        # Store embeddings
        self.product_embeddings = {
            pid: emb for pid, emb in zip(self.product_ids, embeddings_array)
        }
        
        print(f"✓ Built FAISS index with {len(self.product_ids)} products")
        print(f"  Embedding dim: {embedding_dim}")
    
    def set_cooccurrence_matrix(self, cooccur_matrix: Dict[int, Counter]):
        """Set co-occurrence matrix."""
        self.cooccur_matrix = cooccur_matrix
        print(f"✓ Loaded co-occurrence matrix ({len(cooccur_matrix)} products)")
    
    def set_transition_matrix(self, transition_matrix: Dict[int, Counter]):
        """Set transition matrix."""
        self.transition_matrix = transition_matrix
        print(f"✓ Loaded transition matrix ({len(transition_matrix)} products)")
    
    def retrieve_semantic(
        self,
        query_products: List[int],
        k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve candidates using semantic similarity (FAISS).
        
        Args:
            query_products: List of product IDs in session
            k: Number of candidates to retrieve
            
        Returns:
            List of (product_id, score) tuples
        """
        if self.faiss_index is None:
            return []
        
        k = k or self.faiss_k
        
        # Get query embeddings
        query_embeddings = []
        for pid in query_products:
            if pid in self.product_embeddings:
                query_embeddings.append(self.product_embeddings[pid])
        
        if not query_embeddings:
            return []
        
        # Average embeddings
        query_vec = np.mean(query_embeddings, axis=0).reshape(1, -1).astype('float32')
        query_vec = normalize_embeddings(query_vec)
        
        # Search
        scores, indices = self.faiss_index.search(query_vec, k)
        
        # Convert to product IDs
        candidates = []
        for idx, score in zip(indices[0], scores[0]):
            pid = self.product_ids[idx]
            # Exclude items already in session
            if pid not in query_products:
                candidates.append((pid, float(score)))
        
        return candidates
    
    def retrieve_cooccurrence(
        self,
        query_products: List[int],
        k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Retrieve candidates using co-occurrence matrix.
        
        Args:
            query_products: List of product IDs in session
            k: Number of candidates to retrieve
            
        Returns:
            List of (product_id, score) tuples
        """
        if self.cooccur_matrix is None:
            return []
        
        k = k or self.num_candidates
        
        # Aggregate scores
        scores = defaultdict(float)
        
        for pid in query_products:
            if pid in self.cooccur_matrix:
                for candidate, count in self.cooccur_matrix[pid].items():
                    if candidate not in query_products:
                        scores[candidate] += count
        
        # Sort and return top-k
        candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return candidates
    
    def retrieve_transitions(
        self,
        query_products: List[int],
        k: Optional[int] = None,
        last_item_boost: float = 2.0
    ) -> List[Tuple[int, float]]:
        """
        Retrieve candidates using transition matrix.
        
        Args:
            query_products: List of product IDs in session
            k: Number of candidates to retrieve
            last_item_boost: Boost for transitions from last item
            
        Returns:
            List of (product_id, score) tuples
        """
        if self.transition_matrix is None:
            return []
        
        k = k or self.num_candidates
        
        # Aggregate scores with position weighting
        scores = defaultdict(float)
        
        for i, pid in enumerate(query_products):
            if pid in self.transition_matrix:
                # Weight: more recent items get higher weight
                position_weight = 1.0 + (i / len(query_products))
                
                # Extra boost for last item
                if i == len(query_products) - 1:
                    position_weight *= last_item_boost
                
                for candidate, count in self.transition_matrix[pid].items():
                    if candidate not in query_products:
                        scores[candidate] += count * position_weight
        
        # Sort and return top-k
        candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return candidates
    
    def retrieve_vsknn(
        self,
        query_products: List[int],
        all_sessions: List[Dict],
        k: Optional[int] = None,
        num_neighbors: int = 100
    ) -> List[Tuple[int, float]]:
        """
        Variable-Interval Session KNN retrieval.
        
        Args:
            query_products: List of product IDs in current session
            all_sessions: List of all historical sessions
            k: Number of candidates to retrieve
            num_neighbors: Number of similar sessions to consider
            
        Returns:
            List of (product_id, score) tuples
        """
        k = k or self.num_candidates
        
        # Find similar sessions
        query_set = set(query_products)
        session_scores = []
        
        for session in all_sessions:
            session_set = set(session['products'])
            
            # Jaccard similarity
            intersection = len(query_set & session_set)
            union = len(query_set | session_set)
            
            if union > 0:
                similarity = intersection / union
                session_scores.append((session, similarity))
        
        # Get top-K similar sessions
        session_scores.sort(key=lambda x: x[1], reverse=True)
        top_sessions = session_scores[:num_neighbors]
        
        # Aggregate candidate scores from similar sessions
        candidate_scores = defaultdict(float)
        
        for session, sim_score in top_sessions:
            # Weight items by position (later items get higher weight)
            for i, pid in enumerate(session['products']):
                if pid not in query_products:
                    position_weight = (i + 1) / len(session['products'])
                    candidate_scores[pid] += sim_score * position_weight
        
        # Sort and return top-k
        candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return candidates
    
    def retrieve_by_intent(
        self,
        session_slugs: List[str],
        k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve candidates based on category page browsing intent."""
        import pickle
        from src.domain_rules import SLUG_TO_CAT
        try:
            with open('src/popularity_data.pkl', 'rb') as f:
                pop_data = pickle.load(f)
        except:
            return []
        k = k or self.num_candidates
        scores = defaultdict(float)
        found_cat = False
        for slug in session_slugs:
            cat_name = SLUG_TO_CAT.get(slug)
            if cat_name and cat_name in pop_data['category']:
                found_cat = True
                top_items = pop_data['category'][cat_name]
                for i, pid in enumerate(top_items):
                    scores[pid] += (len(top_items) - i) * 1.0
        if not found_cat:
            for i, pid in enumerate(pop_data['global'][:k]):
                scores[pid] += (k - i) * 0.1
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def retrieve_params(
        self,
        query_products: List[int],
        k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve candidates based on parameter similarity."""
        import pickle
        try:
            with open('src/param_neighbors.pkl', 'rb') as f:
                neighbors = pickle.load(f)
        except:
            return []
        k = k or self.num_candidates
        scores = defaultdict(float)
        for pid in query_products:
            if pid in neighbors:
                for target_pid, sim in neighbors[pid]:
                    scores[target_pid] += sim * 1.5
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def retrieve_biological(
        self,
        query_products: List[int],
        k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Retrieve candidates based on biological progression."""
        import pickle
        try:
            with open('src/biological_stages.pkl', 'rb') as f:
                stages = pickle.load(f)
            with open('src/stage_popularity.pkl', 'rb') as f:
                stage_tops = pickle.load(f)
        except:
            return []
        k = k or self.num_candidates
        scores = defaultdict(float)
        for pid in query_products:
            s = stages.get(pid, -1)
            if s != -1:
                # SAME stage
                for i, top_pid in enumerate(stage_tops.get(s, [])):
                    scores[top_pid] += (len(stage_tops[s]) - i) * 1.0
                # NEXT stage
                next_s = s + 1
                if next_s in stage_tops:
                    for i, top_pid in enumerate(stage_tops[next_s]):
                        scores[top_pid] += (len(stage_tops[next_s]) - i) * 1.5
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def retrieve_hybrid(
        self,
        query_products: List[int],
        session_slugs: Optional[List[str]] = None,
        use_semantic: bool = True,
        use_cooccurrence: bool = True,
        use_transitions: bool = True,
        use_intent: bool = True,
        use_biological: bool = True,
        use_params: bool = True,
        all_sessions: Optional[List[Dict]] = None
    ) -> List[Tuple[int, float]]:
        """Combine all retrieval signals into a single candidate list."""
        all_candidates = defaultdict(float)
        conf = self.config['retrieval']
        
        if use_intent and session_slugs:
            for pid, score in self.retrieve_by_intent(session_slugs):
                all_candidates[pid] += score * conf.get('intent_weight', 2.5)

        if use_params and query_products:
            for pid, score in self.retrieve_params(query_products):
                all_candidates[pid] += score * conf.get('parameter_weight', 3.0)

        if use_biological and query_products:
            for pid, score in self.retrieve_biological(query_products):
                all_candidates[pid] += score * conf.get('biological_weight', 3.0)

        if query_products and use_cooccurrence and self.cooccur_matrix is not None:
            max_s = max([s for pid, s in self.retrieve_cooccurrence(query_products)], default=1.0)
            for pid, score in self.retrieve_cooccurrence(query_products):
                all_candidates[pid] += (score / max_s) * conf.get('covisit_weight', 2.0)

        if query_products and use_transitions and self.transition_matrix is not None:
            max_s = max([s for pid, s in self.retrieve_transitions(query_products)], default=1.0)
            for pid, score in self.retrieve_transitions(query_products):
                all_candidates[pid] += (score / max_s) * conf.get('transition_weight', 3.0)

        return sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)[:self.num_candidates]


class ReciproclaRankFusion:
    """Reciprocal Rank Fusion for combining rankings."""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF.
        
        Args:
            k: RRF constant (typically 60)
        """
        self.k = k
    
    def fuse(
        self,
        rankings: List[List[int]],
        weights: Optional[List[float]] = None
    ) -> List[int]:
        """
        Fuse multiple rankings using RRF.
        
        Args:
            rankings: List of rankings (each is list of product IDs)
            weights: Optional weights for each ranking
            
        Returns:
            Fused ranking (list of product IDs)
        """
        if weights is None:
            weights = [1.0] * len(rankings)
        
        scores = defaultdict(float)
        
        for ranking, weight in zip(rankings, weights):
            for rank, item in enumerate(ranking):
                scores[item] += weight / (self.k + rank + 1)
        
        # Sort by score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [item for item, score in fused]


if __name__ == "__main__":
    # Test retriever
    import yaml
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    retriever = CandidateRetriever(config)
    
    # Test with dummy embeddings
    dummy_embeddings = {
        i: np.random.randn(256) for i in range(100)
    }
    
    retriever.build_faiss_index(dummy_embeddings)
    
    # Test retrieval
    query = [1, 5, 10]
    candidates = retriever.retrieve_semantic(query, k=20)
    
    print(f"Retrieved {len(candidates)} candidates")
    print(f"Top 5: {candidates[:5]}")
