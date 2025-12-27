"""
UltimateRecommenderV7_0: The GBDT-Hybrid Professional Engine.
Uses LightGBM (reranker_v5) for high-precision Discovery, 
complemented by a heuristic History layer.
"""
import pandas as pd
import numpy as np
import os
import json
import pickle
import lightgbm as lgb
from collections import defaultdict, Counter
from tqdm.auto import tqdm

# Imports from main.py if available
try:
    from main import (
        PopularityModel, CooccurrenceModel, SequentialTransitionModel, SearchQueryMatcher,
        build_watch_to_visit_mapping, prepare_hits_data, extract_all_session_features,
        Config
    )
except ImportError:
    pass

class UltimateRecommenderV7_0:
    def __init__(self, popularity, mappings):
        self.popularity = popularity
        self.mappings = mappings
        
        # 1. Load Reranker
        print("Loading LGBM Reranker V5...")
        with open('src/reranker_v5.pkl', 'rb') as f:
            self.model = pickle.load(f)
            
        # 2. Load Mapping Data
        print("Loading Metadata...")
        p_df = pd.read_csv('data/new_site_products.csv')
        self.id_to_cat = dict(zip(p_df['id'], p_df['main_category']))
        self.id_to_brand = dict(zip(p_df['id'], p_df['brand']))
        self.global_pop = popularity.global_popularity
        
        # 3. Load Retrieval Data
        with open('src/param_neighbors.pkl', 'rb') as f:
            neighbors = pickle.load(f)
            self.param_sim_map = {pid: {n: s for n, s in ns} for pid, ns in neighbors.items()}
            
        with open('src/city_popularity.pkl', 'rb') as f:
            self.city_pop = pickle.load(f)
            
        with open('src/user_product_history.pkl', 'rb') as f:
            self.user_history = pickle.load(f)
            
        from src.domain_rules import BIOLOGICAL_TRANSITIONS
        self.bio_trans = BIOLOGICAL_TRANSITIONS

    def predict(self, session_features, n=6):
        curr_pids = session_features.get('product_ids', [])
        user_hash = session_features.get('user_hash')
        city = session_features.get('city', 'Unknown')
        
        query_pids = list(dict.fromkeys(curr_pids))
        seen = set()
        recommendations = []
        
        # --- STAGE 1: HEURISTIC HISTORY (Slots 1-3) ---
        # Current Session
        for p in reversed(query_pids):
            if int(p) not in seen:
                recommendations.append(int(p))
                seen.add(int(p))
            if len(recommendations) >= 3: break
            
        # Across-Session (if slots remain)
        if len(recommendations) < 3 and user_hash and str(user_hash) in self.user_history:
            for p in self.user_history[str(user_hash)]:
                if int(p) not in seen:
                    recommendations.append(int(p))
                    seen.add(int(p))
                if len(recommendations) >= 3: break

        # --- STAGE 2: GBDT DISCOVERY (Remaining Slots) ---
        if len(recommendations) < n:
            candidates = set()
            last_pid = query_pids[-1] if query_pids else None
            
            # i. Retrieval (Top 50 unique)
            if last_pid and last_pid in self.param_sim_map:
                candidates.update(list(self.param_sim_map[last_pid].keys())[:20])
            
            candidates.update(self.city_pop.get(city, [])[:20])
            candidates.update(self.global_pop[:20])
            
            candidates = {p for p in candidates if int(p) not in seen}
            
            if candidates:
                # ii. Feature Extraction for Reranker
                X_cand = []
                pids_cand = []
                for cand in candidates:
                    feat = {}
                    feat['same_cat'] = 1.0 if self.id_to_cat.get(cand) == self.id_to_cat.get(last_pid) else 0.0
                    feat['same_brand'] = 1.0 if self.id_to_brand.get(cand) == self.id_to_brand.get(last_pid) else 0.0
                    
                    max_sim = 0
                    if last_pid and last_pid in self.param_sim_map and cand in self.param_sim_map[last_pid]:
                        max_sim = self.param_sim_map[last_pid][cand]
                    feat['param_sim'] = max_sim
                    
                    try: feat['global_rank'] = self.global_pop.index(cand)
                    except: feat['global_rank'] = 999
                    
                    c_list = self.city_pop.get(city, [])
                    try: feat['city_rank'] = c_list.index(cand)
                    except: feat['city_rank'] = 999
                    
                    feat['in_lt_history'] = 1.0 if user_hash and str(user_hash) in self.user_history and int(cand) in self.user_history[str(user_hash)] else 0.0
                    
                    feat['is_bio_successor'] = 1.0 if last_pid in self.bio_trans and cand in self.bio_trans[last_pid] else 0.0
                    
                    X_cand.append(feat)
                    pids_cand.append(cand)
                    
                # iii. Inference
                df_X = pd.DataFrame(X_cand)
                # Ensure column order matches training if possible (handled by LGBM usually)
                probs = self.model.predict_proba(df_X)[:, 1]
                
                # iv. Rank and Pick
                ranked_candidates = sorted(zip(pids_cand, probs), key=lambda x: x[1], reverse=True)
                for p, prob in ranked_candidates:
                    if int(p) not in seen:
                        recommendations.append(int(p))
                        seen.add(int(p))
                    if len(recommendations) >= n: break

        # --- STAGE 3: GLOBAL FALLBACK ---
        if len(recommendations) < n:
            for p in self.global_pop:
                if int(p) not in seen:
                    recommendations.append(int(p))
                    seen.add(int(p))
                if len(recommendations) >= n: break
                
        return recommendations[:n]

if __name__ == "__main__":
    import main
    mappings = main.mappings
    
    recommender = UltimateRecommenderV7_0(main.popularity_model, mappings)
    
    v_test = main.visits_test
    test_watch_to_visit = build_watch_to_visit_mapping(v_test)
    test_hits_prepared = prepare_hits_data(main.hits_test, test_watch_to_visit, mappings['new_slug_to_id'])
    
    # Feature extraction includes city
    v_test_data = v_test[['visit_id', 'counter_user_id_hash', 'region_city']]
    vid_to_user = dict(zip(v_test_data['visit_id'], v_test_data['counter_user_id_hash']))
    vid_to_city = dict(zip(v_test_data['visit_id'], v_test_data['region_city']))
    
    vids = v_test['visit_id'].tolist()
    session_features = extract_all_session_features(test_hits_prepared, vids)
    
    print(f"\nGenerating V7.0 GBDT-Hybrid Submission...")
    predictions = []
    for visit_id in tqdm(vids, desc="Predicting"):
        features = session_features.get(visit_id, {})
        features['user_hash'] = vid_to_user.get(visit_id)
        features['city'] = vid_to_city.get(visit_id, 'Unknown')
            
        recs = recommender.predict(features, n=6)
        
        # Valid ID check
        valid_recs = []
        for r in recs:
            if int(r) in mappings['valid_new_ids']: valid_recs.append(int(r))
        
        while len(valid_recs) < 6:
            for p in main.popularity_model.global_popularity:
                if int(p) not in set(valid_recs) and int(p) in mappings['valid_new_ids']:
                    valid_recs.append(int(p))
                if len(valid_recs) >= 6: break
            break
            
        predictions.append({
            'visit_id': visit_id,
            'product_ids': ' '.join(map(str, valid_recs[:6]))
        })
        
    pd.DataFrame(predictions).to_csv('submission_v7_0.csv', index=False)
    print("✓ Submission V7.0 saved.")