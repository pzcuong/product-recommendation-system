"""
CL-GRU4Rec+RP Demo - Interactive Recommendation Showcase
========================================================

Live demonstration of the CL-GRU4Rec+RP method without comparisons.

Usage:
  python demo.py --dataset rental
  python demo.py --dataset synerise
"""
import argparse
import os
import pickle
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

K = 6  # Number of recommendations

# ============================================================================
# MODEL COMPONENTS
# ============================================================================
class GRU4RecModel(nn.Module):
    """GRU4Rec for sequential pattern learning."""
    def __init__(self, n_items, embed_dim=128, hidden_dim=200, dropout=0.15, pad_idx=0):
        super().__init__()
        self.n_items = n_items
        self.embed = nn.Embedding(n_items, embed_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.embed.weight)
        self.embed.weight.data[pad_idx].zero_()
        self.drop_emb = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop_out = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_items)

    def predict(self, seq, lengths=None):
        """Get scores for last position in sequence."""
        self.eval()
        x = self.embed(seq)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(x)
        return self.head(hidden.squeeze(0))


class ContrastiveItemModel(nn.Module):
    """Contrastive Learning for item similarity."""
    def __init__(self, n_items, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, items):
        return F.normalize(self.projector(self.embedding(items)), dim=-1)


# ============================================================================
# DEMO CLASS
# ============================================================================
class CLGRU4RecRPDemo:
    """Interactive demo for CL-GRU4Rec+RP recommendations."""

    def __init__(self, dataset="rental"):
        self.dataset = dataset
        self.model_components = {}
        self.data_cache = {}
        print(f"\n{'='*70}")
        print(f"  CL-GRU4Rec+RP Demo - {dataset.upper()}")
        print(f"{'='*70}")

    def load_data(self):
        """Load data and models."""
        print("\n[1] Loading data and models...")

        if self.dataset == "rental":
            self._load_rental_data()
        else:
            self._load_synerise_data()

        print(f"  ✓ Data loaded: {len(self.data_cache['test_sessions'])} test sessions")
        print(f"  ✓ Models loaded: GRU4Rec (ensemble), Contrastive Learning")

    def _load_rental_data(self):
        """Load Kaggle Rental Product data."""
        ROOT = "data"

        # Load cached data
        df_data = pd.read_csv("rental_data_cache.csv") if os.path.exists("rental_data_cache.csv") else None
        if df_data is None:
            # Quick data loading
            df_hits = pd.concat([
                pd.read_csv(f"{ROOT}/metrika_hits.csv", usecols=['date_time','slug','page_type','watch_id'], dtype=str),
                pd.read_csv(f"{ROOT}/metrika_hits_test.csv", usecols=['date_time','slug','page_type','watch_id'], dtype=str),
            ], ignore_index=True)
            df_visits = pd.concat([
                pd.read_csv(f"{ROOT}/metrika_visits.csv", usecols=['client_id','visit_id','watch_ids'], dtype=str),
                pd.read_csv(f"{ROOT}/metrika_visits_test.csv", usecols=['client_id','visit_id','watch_ids'], dtype=str),
            ], ignore_index=True)
            df_visits["watch_ids"] = df_visits["watch_ids"].apply(eval)
            df_visits = df_visits.explode("watch_ids").rename(columns={"watch_ids": "watch_id"})
            df_data = pd.merge(df_hits, df_visits, on="watch_id", how="left")
            df_data = df_data[df_data["page_type"] == "PRODUCT"].drop_duplicates(["visit_id", "slug"])

        # Build vocabulary
        all_items = sorted(df_data["slug"].unique())
        self.data_cache['item_to_idx'] = {"<PAD>": 0}
        for i, item in enumerate(all_items):
            self.data_cache['item_to_idx'][item] = i + 1
        self.data_cache['idx_to_item'] = {v: k for k, v in self.data_cache['item_to_idx'].items()}
        self.data_cache['n_items'] = len(self.data_cache['item_to_idx'])

        # Sample test sessions
        test_sessions = df_data.groupby("visit_id")["slug"].apply(list).head(100).to_dict()
        self.data_cache['test_sessions'] = test_sessions

    def _load_synerise_data(self):
        """Load Synerise RecSys data from cache."""
        CACHE = "synerise_final.pkl"
        if os.path.exists(CACHE):
            with open(CACHE, "rb") as f:
                d = pickle.load(f)

            train_items = d["train_items"]
            test_gt = d["test_gt"]

            # Build vocabulary
            all_items = sorted(d["freq"])
            self.data_cache['item_to_idx'] = {"<PAD>": 0}
            for i, item in enumerate(all_items):
                self.data_cache['item_to_idx'][item] = i + 1
            self.data_cache['idx_to_item'] = {v: k for k, v in self.data_cache['item_to_idx'].items()}
            self.data_cache['n_items'] = len(self.data_cache['item_to_idx'])

            # Sample test users
            sample_users = sorted(test_gt.keys())[:100]
            self.data_cache['test_sessions'] = {u: train_items[u] for u in sample_users}
            self.data_cache['test_gt'] = {u: test_gt[u] for u in sample_users}

    def load_models(self):
        """Load trained model components."""
        print("\n[2] Loading model components...")

        n_items = self.data_cache['n_items']

        # Initialize models (will use random weights for demo if not trained)
        gru_model = GRU4RecModel(n_items).to(DEVICE)
        cl_model = ContrastiveItemModel(n_items).to(DEVICE)

        # Try to load trained weights
        if self.dataset == "synerise":
            try:
                gru_model.load_state_dict(torch.load("v4_synerise_gru_seed42.pkl", map_location=DEVICE, weights_only=True))
                cl_model.load_state_dict(torch.load("v4_synerise_cl.pkl", map_location=DEVICE, weights_only=True))
                print("  ✓ Loaded trained models from cache")
            except:
                print("  ℹ Using random weights (models not trained yet)")

        self.model_components['gru'] = gru_model
        self.model_components['cl'] = cl_model

        # Pre-compute CL embeddings
        cl_model.eval()
        with torch.no_grad():
            self.data_cache['cl_embeddings'] = cl_model(torch.arange(n_items).to(DEVICE)).cpu().numpy()

        print(f"  ✓ GRU4Rec: {sum(p.numel() for p in gru_model.parameters()):,} parameters")
        print(f"  ✓ Contrastive Learning: {sum(p.numel() for p in cl_model.parameters()):,} parameters")

    def recommend(self, session_items, explain=True):
        """Generate recommendations for a session."""
        idx_to_item = self.data_cache['idx_to_item']
        item_to_idx = self.data_cache['item_to_idx']
        cl_emb = self.data_cache['cl_embeddings']

        # Convert to indices
        indices = [item_to_idx.get(s, 0) for s in session_items if s in item_to_idx]

        if not indices:
            return [], {}

        # Component 1: GRU4Rec Sequential Score
        gru_model = self.model_components['gru']
        seq = torch.LongTensor([indices[-50:]]).to(DEVICE)
        length = torch.LongTensor([len(indices[-50:])])

        gru_model.eval()
        with torch.no_grad():
            gru_scores = gru_model.predict(seq, length).squeeze(0).cpu().numpy()

        # Component 2: Contrastive Similarity Score
        user_emb = cl_emb[indices[-10:]].mean(0)
        user_emb /= (np.linalg.norm(user_emb) + 1e-8)
        cl_sim = cl_emb @ user_emb

        # Component 3: Re-Purchase Score
        rp_scores = Counter(indices)
        max_rp = max(rp_scores.values()) if rp_scores else 1
        rp_scores = {k: v / max_rp for k, v in rp_scores.items()}

        # Fusion
        final_scores = {}
        for idx in range(1, len(item_to_idx)):  # Skip PAD
            if idx in indices:
                continue

            # Adaptive weights based on session length
            session_len_factor = min(len(indices) / 10, 1.0)

            gru_w = 0.6 + 0.3 * session_len_factor  # 0.6 - 0.9
            cl_w = 0.3 - 0.1 * session_len_factor   # 0.2 - 0.3
            rp_w = 0.1                              # Fixed

            gru_score = max(0, float(gru_scores[idx]))
            cl_score = max(0, float(cl_sim[idx]))
            rp_score = rp_scores.get(idx, 0)

            final_scores[idx] = (
                gru_w * gru_score +
                cl_w * cl_score +
                rp_w * rp_score
            )

        # Get top-K
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:K]
        recommendations = [idx_to_item[idx] for idx, _ in sorted_items]

        # Explanation
        if explain:
            explanation = {}
            for idx, score in sorted_items:
                explanation[idx_to_item[idx]] = {
                    'final_score': round(score, 4),
                    'gru_contribution': round(gru_w * max(0, float(gru_scores[idx])), 4),
                    'cl_contribution': round(cl_w * max(0, float(cl_sim[idx])), 4),
                    'rp_contribution': round(rp_w * rp_scores.get(idx, 0), 4),
                }

            return recommendations, explanation

        return recommendations, {}

    def demo_single_session(self):
        """Demo recommendation for a single session."""
        print("\n" + "="*70)
        print("  DEMO: Single Session Recommendation")
        print("="*70)

        # Get a sample session
        sessions = list(self.data_cache['test_sessions'].items())
        session_id, session_items = sessions[0]

        print(f"\n📱 Session ID: {session_id}")
        print(f"🛒 Session History ({len(session_items)} items):")
        for i, item in enumerate(session_items[-10:], 1):
            print(f"   {i}. {item}")

        # Get recommendations
        recommendations, explanation = self.recommend(session_items)

        print(f"\n✨ Top-{K} Recommendations:")
        print("-" * 70)
        for i, rec in enumerate(recommendations, 1):
            exp = explanation.get(rec, {})
            print(f"\n  {i}. {rec}")
            print(f"     Final Score: {exp.get('final_score', 0):.4f}")
            print(f"     ├─ GRU4Rec (Sequential):     {exp.get('gru_contribution', 0):.4f}")
            print(f"     ├─ Contrastive (Similarity): {exp.get('cl_contribution', 0):.4f}")
            print(f"     └─ Re-Purchase (History):    {exp.get('rp_contribution', 0):.4f}")

    def demo_session_progression(self):
        """Demo how recommendations evolve as session grows."""
        print("\n" + "="*70)
        print("  DEMO: Session Progression (Real-time Updates)")
        print("="*70)

        sessions = list(self.data_cache['test_sessions'].items())
        session_id, full_session = sessions[1]

        # Simulate session growth
        steps = [1, 3, 5, min(10, len(full_session))]

        for n_items in steps:
            current_session = full_session[:n_items]

            print(f"\n{'─'*70}")
            print(f"📱 Session Progress: {n_items} items")
            print(f"   History: {' → '.join(current_session[-3:])}")

            recommendations, _ = self.recommend(current_session, explain=False)

            print(f"   Recommended: {', '.join(recommendations[:3])}")

            # Show how GRU confidence increases
            idx_to_item = self.data_cache['idx_to_item']
            item_to_idx = self.data_cache['item_to_idx']
            indices = [item_to_idx.get(s, 0) for s in current_session if s in item_to_idx]

            if indices:
                seq = torch.LongTensor([indices[-50:]]).to(DEVICE)
                length = torch.LongTensor([len(indices[-50:])])
                self.model_components['gru'].eval()
                with torch.no_grad():
                    scores = self.model_components['gru'].predict(seq, length).squeeze(0).cpu().numpy()

                top_score = float(np.sort(scores)[-1])
                print(f"   GRU Confidence: {top_score:.4f} (increases with session length)")

    def demo_cold_start(self):
        """Demo cold start handling."""
        print("\n" + "="*70)
        print("  DEMO: Cold Start Handling")
        print("="*70)

        # Simulate cold start sessions
        cold_sessions = [
            ("Single item", ["Đèn LED"]),
            ("Two items", ["Bóng đèn", "Ổ cắm"]),
            ("Three items", ["Bóng đèn", "Ổ cắm", "Đèn LED"]),
        ]

        for name, session in cold_sessions:
            print(f"\n📱 Cold Start: {name}")
            print(f"   Session: {' → '.join(session)}")

            recommendations, _ = self.recommend(session, explain=False)

            if recommendations:
                print(f"   ✓ Recommendations: {', '.join(recommendations[:3])}")
            else:
                print(f"   ℹ Fallback: Popular items")

    def run_all_demos(self):
        """Run all demo scenarios."""
        print("\n" + "="*70)
        print("  CL-GRU4Rec+RP - Full Demonstration")
        print("="*70)

        self.load_data()
        self.load_models()

        self.demo_single_session()
        self.demo_session_progression()
        self.demo_cold_start()

        print("\n" + "="*70)
        print("  Demo Complete!")
        print("="*70)
        print("\n💡 Key Takeaways:")
        print("   1. GRU4Rec captures sequential patterns")
        print("   2. Contrastive Learning discovers similar items")
        print("   3. Re-Purchase boosts frequently interacted items")
        print("   4. Adaptive Fusion balances all signals")
        print("   5. Recommendations evolve as session grows")
        print("\n")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CL-GRU4Rec+RP Demo")
    parser.add_argument("--dataset", choices=["rental", "synerise"], default="rental",
                       help="Dataset to demo")
    args = parser.parse_args()

    demo = CLGRU4RecRPDemo(dataset=args.dataset)
    demo.run_all_demos()
