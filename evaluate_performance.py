"""
Local Validation for V7.0 (GBDT-Hybrid).
Uses "Filtered Duplicates" logic to correlate with Kaggle.
"""
import pandas as pd
import json
import pickle
from collections import defaultdict, Counter
from tqdm.auto import tqdm
from generate_submission import UltimateRecommenderV7_0
import main

def evaluate_v7():
    print("Preparing Validation Data...")
    hits = pd.read_csv('data/metrika_hits.csv', usecols=['project_id', 'watch_id', 'slug', 'date_time', 'counter_user_id_hash'])
    hits = hits[hits['project_id'] == 0]
    slug_to_id = dict(pd.read_csv('data/new_site_products.csv')[['slug', 'id']].dropna().values)
    hits['pid'] = hits['slug'].map(slug_to_id)
    hits = hits.dropna(subset=['pid']).sort_values(['watch_id', 'date_time'])

    v_df = pd.read_csv('data/metrika_visits.csv', usecols=['visit_id', 'watch_ids', 'region_city'])
    w2v = {}
    vid_to_city = {}
    for _, row in v_df.iterrows():
        try:
            wids = json.loads(row['watch_ids'])
            for wid in wids: 
                w2v[int(wid)] = row['visit_id']
                vid_to_city[row['visit_id']] = row['region_city']
        except: pass
    hits['visit_id'] = hits['watch_id'].map(w2v)
    hits = hits.dropna(subset=['visit_id']).sort_values(['visit_id', 'date_time'])

    val_hits = hits[hits['date_time'] >= '2025-08-01']
    
    # Initialize V7.0
    mappings = main.mappings
    recommender = UltimateRecommenderV7_0(main.popularity_model, mappings)
    
    hits_count = 0
    total = 0

    print("Evaluating V7.0 GBDT-Hybrid...")
    for vid, group in tqdm(val_hits.groupby('visit_id')):
        pids = group['pid'].tolist()
        user = group['counter_user_id_hash'].iloc[0]
        city = vid_to_city.get(vid, 'Unknown')
        
        # Filter contiguous duplicates
        unique_pids = []
        for p in pids:
            if not unique_pids or p != unique_pids[-1]:
                unique_pids.append(int(p))
        
        if len(unique_pids) < 2: continue
        
        # Prefix = first 50%, Targets = rest
        mid = len(unique_pids) // 2
        if mid == 0: mid = 1
        prefix = unique_pids[:mid]
        targets = set(unique_pids[mid:])
        
        # Predict
        features = {
            'product_ids': prefix,
            'user_hash': user,
            'city': city
        }
        recs = recommender.predict(features, n=6)
        
        total += 1
        if any(int(r) in targets for r in recs):
            hits_count += 1

    score = hits_count / total if total > 0 else 0
    print(f"\nLOCAL VALIDATION (Filtered):")
    print(f"Total Sessions: {total}")
    print(f"V7.0 Recall@6: {score:.4f}")
    return score

if __name__ == "__main__":
    evaluate_v7()
