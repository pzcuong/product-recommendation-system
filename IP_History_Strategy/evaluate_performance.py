"""
Local Validation for V7.0 (GBDT-Hybrid).
Uses "Filtered Duplicates" logic to correlate with Kaggle.
"""
import pandas as pd
import json
import pickle
from collections import defaultdict, Counter
from tqdm.auto import tqdm
import numpy as np
from generate_submission import UltimateRecommenderV7_0
import main

def evaluate_v7():
    print("Preparing Validation Data...")
    hits = pd.read_csv('data/metrika_hits.csv', 
                      usecols=['project_id', 'watch_id', 'slug', 'date_time', 'counter_user_id_hash', 'page_type', 'ip_address'])
    hits = hits[hits['project_id'] == 0]
    
    # Extract Search Queries per Visit
    print("Extracting Search Queries...")
    
    # Load Visits for mapping
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
        
    # Map hits to visits
    hits['visit_id'] = hits['watch_id'].map(w2v)
    hits = hits.dropna(subset=['visit_id']).sort_values(['visit_id', 'date_time'])
    
    # Build Visit -> Queries Map
    from urllib.parse import urlparse, parse_qs
    vid_to_queries = defaultdict(list)
    search_mask = (hits['page_type'] == 'SEARCH')
    
    for vid, group in hits[search_mask].groupby('visit_id'):
        queries = []
        for url in group['slug'].dropna():
            try:
                # Handle potential full URLs or relative paths
                if 'http' not in url: url = 'http://dummy.com' + url
                parsed = urlparse(str(url))
                q_list = parse_qs(parsed.query).get('q', [])
                if q_list:
                    queries.append(q_list[0])
            except:
                pass
        if queries:
            vid_to_queries[vid] = list(set(queries)) # Unique queries per session

    # Continue with Product ID mapping
    slug_to_id = dict(pd.read_csv('data/new_site_products.csv')[['slug', 'id']].dropna().values)
    hits['pid'] = hits['slug'].map(slug_to_id)
    
    # Validation Grid
    val_hits = hits[hits['date_time'] >= '2025-08-01']
    val_hits = val_hits.dropna(subset=['pid']) # Only product hits for targets
    
    # Initialize V7.0
    mappings = main.mappings
    recommender = UltimateRecommenderV7_0(main.popularity_model, mappings)
    
    print("Evaluating V7.0 GBDT-Hybrid with Simulated Cold Sessions...")
    cold_total = 0
    cold_recall = 0
    warm_total = 0
    warm_recall = 0
    total = 0
    
    for vid, group in tqdm(val_hits.groupby('visit_id')):
        pids = group['pid'].tolist()
        user = group['counter_user_id_hash'].iloc[0]
        ip = group['ip_address'].iloc[0]
        city = vid_to_city.get(vid, 'Unknown')
        queries = vid_to_queries.get(vid, [])
        
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
        
        # --- PREDICTION 1: WARM (Actual User) ---
        features_warm = {
            'product_ids': prefix,
            'user_hash': user,
            'ip_address': ip,
            'city': city,
            'search_queries': queries
        }
        recs_warm = recommender.predict(features_warm, n=6)
        
        hits_warm = len(targets.intersection(recs_warm))
        recall_warm = hits_warm / len(targets)
        
        # --- PREDICTION 2: SIMULATED COLD (Unknown User) ---
        features_cold = {
            'product_ids': prefix,
            'user_hash': None, # Force cold
            'ip_address': ip, # Keep IP (Simulate same household/proxy)
            'city': city,
            'search_queries': queries
        }
        recs_cold = recommender.predict(features_cold, n=6)
        hits_cold = len(targets.intersection(recs_cold))
        recall_cold = hits_cold / len(targets)
        
        # Aggregation
        warm_total += 1
        warm_recall += recall_warm
        
        cold_total += 1
        cold_recall += recall_cold
        
        total += 1

    avg_warm = warm_recall / warm_total if warm_total > 0 else 0
    avg_cold = cold_recall / cold_total if cold_total > 0 else 0
    
    avg_warm = warm_recall / warm_total if warm_total > 0 else 0
    avg_cold = cold_recall / cold_total if cold_total > 0 else 0
    
    # Blended Score (Assuming 50/50 split like Kaggle)
    blended_score = (avg_warm + avg_cold) / 2
    
    print(f"\nLOCAL VALIDATION (Simulated Warm/Cold):")
    print(f"Total Sessions Evaluated: {total}")
    print(f"-"*30)
    print(f"Warm Recall@6: {avg_warm:.4f} (History Available)")
    print(f"Cold Recall@6: {avg_cold:.4f} (Simulated New User)")
    print(f"-"*30)
    print(f"Blended Score (50/50): {blended_score:.4f}")
    
    return blended_score

if __name__ == "__main__":
    evaluate_v7()
