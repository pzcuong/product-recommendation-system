
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm

def build_ip_history():
    print("Loading Hits...")
    # Load New Site hits only (project_id=0) to ensure relevance
    # Load only necessary columns
    hits = pd.read_csv('data/metrika_hits.csv', 
                      usecols=['ip_address', 'slug', 'date_time', 'project_id'])
    
    # Filter New Site
    hits = hits[hits['project_id'] == 0]
    
    # Load Product Map
    new_products = pd.read_csv('data/new_site_products.csv', usecols=['slug', 'id'])
    slug_to_id = dict(zip(new_products['slug'], new_products['id']))
    
    # Map IDs
    hits['pid'] = hits['slug'].map(slug_to_id)
    hits = hits.dropna(subset=['pid'])
    hits['pid'] = hits['pid'].astype(int)
    
    # Sort by time
    print("Sorting by time...")
    hits['date_time'] = pd.to_datetime(hits['date_time'], format='mixed')
    hits = hits.sort_values('date_time')
    
    print("Grouping by IP...")
    # We want the LAST few items viewed by this IP
    # Groupby is expensive on 1M rows? No, it's fine.
    
    ip_history = defaultdict(list)
    
    # Iterate is slow. Vectorized?
    # We can group by IP and aggregate into list
    # But we want to maintain order.
    
    # Optimized approach:
    # 1. Group by IP -> Apply list
    # 2. Convert to dict
    
    # To save memory, we can limit list size during aggregation?
    # Pandas groupby apply is slow.
    # Let's iterate over sorted rows (it's 1.7M rows, takes ~10s in pure python loop if optimized)
    
    # Actually, pandas groupby agg list is decent.
    
    df_grouped = hits.groupby('ip_address')['pid'].apply(list)
    
    # Convert to dict and slice last 20 items (we don't need infinite history)
    print("Building Dictionary...")
    for ip, pids in tqdm(df_grouped.items(), total=len(df_grouped)):
        # Keep last 10 items
        ip_history[ip] = pids[-10:]
        
    print(f"Built History for {len(ip_history)} IPs.")
    
    with open('src/ip_history.pkl', 'wb') as f:
        pickle.dump(dict(ip_history), f)
    print("Saved to src/ip_history.pkl")

if __name__ == "__main__":
    build_ip_history()
