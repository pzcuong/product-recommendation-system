# %% [markdown]
# # Title:
# 
# # Rental Product Recommendation System: Session-Based Next-Item Prediction

# %% [markdown]
# # Summary:
# 
# This competition focuses on predicting the next product page a user will visit during a browsing session on an online rental marketplace for children’s products (strollers, car seats, toys, etc.). The goal is to increase engagement and daily orders by recommending relevant products.
# 
# The dataset includes:
# 
# New site products and orders
# 
# Historical old site products, carts, and orders
# 
# User browsing sessions and hits
# 
# The task is session-based next-item prediction, evaluated using Recall@6, requiring the model to predict exactly 6 product IDs for each test session.
# 
# A baseline solution using top-6 most popular products per session was implemented, and session sequences were aggregated for potential sequential modeling.

# %% [code] {"execution":{"iopub.status.busy":"2025-12-11T15:51:32.951674Z","iopub.execute_input":"2025-12-11T15:51:32.953318Z","iopub.status.idle":"2025-12-11T15:51:35.786955Z","shell.execute_reply.started":"2025-12-11T15:51:32.953285Z","shell.execute_reply":"2025-12-11T15:51:35.785565Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-12-23T19:20:13.277448Z","iopub.execute_input":"2025-12-23T19:20:13.277849Z","iopub.status.idle":"2025-12-23T19:22:41.506062Z","shell.execute_reply.started":"2025-12-23T19:20:13.277815Z","shell.execute_reply":"2025-12-23T19:22:41.504917Z"}}
# =====================================================================
# Rental Product Recommendation System - Complete Notebook
# =====================================================================

# ==============================
# 1. SETUP AND IMPORTS
# ==============================
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import ast
import re
import json
import warnings
from tqdm.auto import tqdm
import gc
import os
from urllib.parse import unquote

warnings.filterwarnings('ignore')

# ==============================
# 2. CONFIGURATION
# ==============================
class Config:
    """Central configuration for the recommendation system."""
    # Auto-detect data path
    if os.path.exists('/kaggle/input/rental-product-recommendation-system/'):
        DATA_PATH = '/kaggle/input/rental-product-recommendation-system/'
    elif os.path.exists('data/'):
        DATA_PATH = 'data/'
    else:
        DATA_PATH = '/kaggle/input/rental-product-recommendation-system/'  # Default
    COOCCURRENCE_WEIGHT = 1.0
    COPURCHASE_WEIGHT = 2.0
    OLD_SITE_DISCOUNT = 0.5
    NUM_PREDICTIONS = 6
    LAST_ITEM_BOOST = 2.0
    SEED = 42

config = Config()
np.random.seed(config.SEED)

# ==============================
# 3. CATEGORY MAPPINGS
# ==============================
CATEGORY_SLUG_TO_RUSSIAN = {
    'kolyaski': ['Коляски', 'Прогулочные коляски'],
    'kolyaski-yoyo': ['Коляски YoYo'],
    'kolyaski-dlya-puteshestviy': ['Коляски для путешествий'],
    'kolyaski-dlya-novorozhdennyh-lyulki': ['Коляски для новорожденных (люльки)'],
    'kolyaski-avtokresla': ['Коляски-автокресла'],
    'progulochnye-kolyaski': ['Прогулочные коляски'],
    'avtokresla-avtolyulki': ['Автокресла, автолюльки'],
    'avtokresla-dlya-novorozhdyonnyh': ['Автокресла для новорождённых'],
    'avtokresla-9-36-kg': ['Автокресла 9-36 кг'],
    'avtokresla-britax-romer': ['Автокресла Britax Romer'],
    'krovatki-manezhi': ['Кроватки, манежи'],
    'kokony-dlya-novorozhdennyh': ['Коконы для новорожденных'],
    'stulchiki-dlya-kormleniya': ['Стульчики для кормления'],
    'kacheli-shezlongi': ['Качели, шезлонги', 'Электрокачели', 'Шезлонги'],
    'elektro-kacheli': ['Электрокачели'],
    'shezlongi': ['Шезлонги'],
    'igrovye-tsentry-i-kompleksy': ['Игровые центры и комплексы'],
    'razvivayuschie-igrushki': ['Развивающие игрушки', 'Музыкальные игрушки', 'Сортеры и пирамидки'],
    'bizibordy': ['Бизиборды', 'Игровые панели и бизиборды'],
    'igrovye-paneli': ['Игровые панели и бизиборды'],
    'igrushki-dlya-vannoy': ['Игрушки для ванной'],
    'begovely': ['Беговелы'],
    'velosipedy': ['Велосипеды'],
    'samokaty': ['Самокаты'],
    'velokresla': ['Велокресла'],
    'katalki': ['Каталки'],
    'kachalki': ['Качалки'],
    'hodunki': ['Классические ходунки', 'Ходунки-каталки'],
    'hodunki-katalki': ['Ходунки-каталки'],
    'klassicheskie-hodunki': ['Классические ходунки'],
    'gorki': ['Горки'],
    'batuty': ['Батуты'],
    'palatki-i-domiki': ['Палатки и домики'],
    'ergoryukzaki': ['Эргорюкзаки'],
    'videonyani': ['Видеоняни'],
    'vesy': ['Весы', 'Весы Саша'],
    'meditsinskie-tovary': ['Медицинские товары'],
    'vannochki-dlya-kupaniya': ['Ванночки для купания'],
    'mashinki-i-garazhi': ['Машинки и гаражи'],
    'parkovki-i-garazhi': ['Машинки и гаражи'],
    'kuhni-i-supermarkety': ['Кухни и супермаркеты'],
    'kuhni-i-domiki': ['Кухни и супермаркеты'],
    'konstruktory': ['Конструкторы'],
    'muzykalnye-igrushki': ['Музыкальные игрушки', 'Музыкальные инструменты'],
    'muzykalnye-instrumenty': ['Музыкальные инструменты'],
    'mobili-i-nochniki': ['Мобили и ночники'],
    'aksessuary-k-sportkompleksam': ['Аксессуары к спорткомплексам'],
    'sportivnye-kompleksy': ['Спортивные комплексы'],
    'chemodany-i-ryukzaki': ['Чемоданы и рюкзаки'],
    '4moms': ['Электрокачели'],
    'ROOT': [],
}

# ==============================
# 4. DATA LOADING
# ==============================
print("Loading data...")
DATA_PATH = config.DATA_PATH

# Core behavioral data
visits = pd.read_csv(f"{DATA_PATH}metrika_visits.csv")
hits = pd.read_csv(f"{DATA_PATH}metrika_hits.csv", low_memory=False)
visits_test = pd.read_csv(f"{DATA_PATH}metrika_visits_test.csv")
hits_test = pd.read_csv(f"{DATA_PATH}metrika_hits_test.csv", low_memory=False)

# Product catalogs
new_products = pd.read_csv(f"{DATA_PATH}new_site_products.csv")
old_products = pd.read_csv(f"{DATA_PATH}old_site_products.csv")
product_mapping = pd.read_csv(f"{DATA_PATH}old_site_new_site_products.csv")

# Order data
new_orders = pd.read_csv(f"{DATA_PATH}new_site_orders.csv")
old_orders = pd.read_csv(f"{DATA_PATH}old_site_orders.csv")
old_carts = pd.read_csv(f"{DATA_PATH}old_site_carts.csv")

print("\n✓ Data loaded successfully!")

# ==============================
# 5. CREATE PRODUCT MAPPINGS
# ==============================
print("Creating product mappings...")
mappings = {
    'new_slug_to_id': dict(zip(new_products['slug'], new_products['id'])),
    'new_id_to_slug': dict(zip(new_products['id'], new_products['slug'])),
    'old_slug_to_id': dict(zip(old_products['slug'], old_products['id'])),
    'old_to_new_id': dict(zip(product_mapping['old_site_id'], product_mapping['new_site_id'])),
    'valid_new_ids': set(new_products['id'].tolist()),
}

# Old slug -> New ID mapping
mappings['old_slug_to_new_id'] = {}
for old_slug, old_id in mappings['old_slug_to_id'].items():
    if old_id in mappings['old_to_new_id']:
        mappings['old_slug_to_new_id'][old_slug] = mappings['old_to_new_id'][old_id]

print(f"  New site products: {len(mappings['new_slug_to_id']):,}")
print(f"  Old products mappable to new: {len(mappings['old_slug_to_new_id']):,}")

# ==============================
# 6. CREATE CATEGORY->PRODUCT MAPPING
# ==============================
print("Creating category -> products mapping...")
category_to_products = defaultdict(list)
for _, row in new_products.iterrows():
    product_id = row['id']
    main_cat = row['main_category']
    if pd.notna(main_cat):
        category_to_products[main_cat].append(product_id)
    if pd.notna(row['categories']):
        try:
            cats = ast.literal_eval(row['categories'])
            for cat in cats:
                if cat != main_cat:
                    category_to_products[cat].append(product_id)
        except:
            pass
category_to_products = dict(category_to_products)
print(f"  Mapped {len(category_to_products)} Russian categories")

# Build product -> main_category mapping
product_to_category = {}
for _, row in new_products.iterrows():
    pid = row['id']
    main_cat = row['main_category']
    if pd.notna(main_cat):
        product_to_category[pid] = main_cat
mappings['product_to_category'] = product_to_category
mappings['category_to_products'] = category_to_products
print(f"  Products with category: {len(product_to_category)}")

# Build order counts (Golden Labels)
order_counts = new_orders['product_id'].value_counts().to_dict()
mappings['order_counts'] = order_counts
print(f"  Products with orders: {len(order_counts)} (total: {sum(order_counts.values())})")

# Build category popularity for category expansion
cat_popularity = defaultdict(Counter)
for pid, cat in product_to_category.items():
    if pid in order_counts:
        cat_popularity[cat][pid] += order_counts[pid]
mappings['cat_popularity'] = dict(cat_popularity)
print(f"  Categories with popularity data: {len(cat_popularity)}")

# Build CO-PURCHASE patterns from OLD orders (products bought together)
print("Building co-purchase patterns from old orders...")
old_order_products = old_orders.groupby('id')['product_id'].apply(list).tolist()
old_multi_orders = [o for o in old_order_products if len(o) > 1]

copurchase = defaultdict(Counter)
for prods in old_multi_orders:
    # Map old product IDs to new
    mapped = [int(mappings['old_to_new_id'].get(p)) for p in prods if p in mappings['old_to_new_id']]
    if len(mapped) > 1:
        for i, p1 in enumerate(mapped):
            for p2 in mapped[i+1:]:
                copurchase[p1][p2] += 1
                copurchase[p2][p1] += 1

mappings['copurchase'] = dict(copurchase)
print(f"  Co-purchase products: {len(copurchase)} from {len(old_multi_orders)} multi-product orders")

# Build CROSS-CATEGORY transition patterns (for cross-category boost)
# This will be populated later during training when we have session data
mappings['cat_transitions'] = defaultdict(Counter)
mappings['cat_product_popularity'] = defaultdict(lambda: defaultdict(Counter))


# ==============================
# 7. UTILITY FUNCTIONS
# ==============================
def build_watch_to_visit_mapping(visits_df: pd.DataFrame) -> Dict[int, int]:
    """Build mapping from watch_id (event) to visit_id (session)."""
    watch_to_visit = {}
    for _, row in tqdm(visits_df.iterrows(), total=len(visits_df), desc="Building watch->visit mapping"):
        visit_id = row['visit_id']
        try:
            watch_ids = ast.literal_eval(str(row['watch_ids']))
            for wid in watch_ids:
                watch_to_visit[int(wid)] = visit_id
        except:
            pass
    return watch_to_visit

def parse_datetime(dt_series: pd.Series) -> pd.Series:
    """Parse datetime with flexible format handling."""
    return pd.to_datetime(dt_series, format='ISO8601', errors='coerce')

def prepare_hits_data(hits: pd.DataFrame, watch_to_visit: Dict[int, int], slug_to_id: Dict[str, int],
                     is_old_site: bool = False, old_slug_to_new_id: Dict[str, int] = None) -> pd.DataFrame:
    """Prepare hits data with visit_id and product_id mappings."""
    hits = hits.copy()
    hits['visit_id'] = hits['watch_id'].map(watch_to_visit)
    hits['date_time'] = parse_datetime(hits['date_time'])
    if is_old_site and old_slug_to_new_id:
        hits['product_id'] = hits['slug'].map(old_slug_to_new_id)
    else:
        hits['product_id'] = hits['slug'].map(slug_to_id)
    hits = hits.sort_values(['visit_id', 'date_time'])
    return hits

def extract_session_features(session_hits: pd.DataFrame) -> Dict:
    """Extract features from a single session's hits."""
    features = {
        'product_ids': [], 'product_slugs': [], 'category_slugs': [],
        'page_sequence': [], 'search_queries': [],
        'has_cart': False, 'has_checkout': False, 'has_order': False,
        'session_length': len(session_hits),
    }
    for _, hit in session_hits.iterrows():
        page_type = hit['page_type']
        slug = hit['slug']
        features['page_sequence'].append(page_type)
        if page_type == 'PRODUCT':
            if pd.notna(slug):
                features['product_slugs'].append(slug)
            if pd.notna(hit.get('product_id')):
                features['product_ids'].append(int(hit['product_id']))
        elif page_type == 'CATEGORY':
            if pd.notna(slug):
                features['category_slugs'].append(slug)
        elif page_type == 'SEARCH':
            url = hit.get('url', '')
            if pd.notna(url):
                match = re.search(r'[?&]q=([^&]+)', str(url))
                if match:
                    try:
                        query = unquote(match.group(1))
                        features['search_queries'].append(query)
                    except:
                        pass
        elif page_type == 'CART':
            features['has_cart'] = True
        elif page_type == 'CHECKOUT':
            features['has_checkout'] = True
        elif page_type == 'ORDER':
            features['has_order'] = True
    return features

def extract_all_session_features(hits: pd.DataFrame, visit_ids: List[int]) -> Dict[int, Dict]:
    """Extract features for all sessions."""
    session_features = {}
    grouped = hits.groupby('visit_id')
    for visit_id in tqdm(visit_ids, desc="Extracting session features"):
        if visit_id in grouped.groups:
            session_hits = grouped.get_group(visit_id)
            session_features[visit_id] = extract_session_features(session_hits)
        else:
            session_features[visit_id] = {
                'product_ids': [], 'product_slugs': [], 'category_slugs': [],
                'page_sequence': [], 'search_queries': [],
                'has_cart': False, 'has_checkout': False, 'has_order': False,
                'session_length': 0,
            }
    return session_features

# ==============================
# 8. MODELS
# ==============================
class PopularityModel:
    """Global and category-specific popularity model."""
    def __init__(self):
        self.global_popularity = []
        self.category_popularity = defaultdict(list)
        self.product_counts = None
    
    def fit(self, hits: pd.DataFrame, category_slug_mapping: Dict[str, List[str]],
            category_to_products: Dict[str, List[int]]):
        """Learn popularity from product view counts."""
        print("Fitting popularity model...")
        product_hits = hits[hits['page_type'] == 'PRODUCT']
        product_counts = product_hits['product_id'].value_counts()
        self.global_popularity = product_counts.index.tolist()
        self.product_counts = product_counts
        print(f"  Global popularity: {len(self.global_popularity)} products ranked")
        
        category_product_views = defaultdict(Counter)
        grouped = hits.groupby('visit_id')
        for visit_id, group in tqdm(grouped, desc="  Building category popularity"):
            categories = group[group['page_type'] == 'CATEGORY']['slug'].dropna().unique()
            products = group[group['page_type'] == 'PRODUCT']['product_id'].dropna().unique()
            for cat in categories:
                for prod in products:
                    category_product_views[cat][int(prod)] += 1
        
        for cat_slug, product_counter in category_product_views.items():
            self.category_popularity[cat_slug] = [p for p, _ in product_counter.most_common(100)]
        
        for cat_slug, russian_cats in category_slug_mapping.items():
            if cat_slug not in self.category_popularity:
                products_in_cat = []
                for rus_cat in russian_cats:
                    if rus_cat in category_to_products:
                        products_in_cat.extend(category_to_products[rus_cat])
                if products_in_cat:
                    products_in_cat = sorted(set(products_in_cat),
                                           key=lambda p: product_counts.get(p, 0), reverse=True)
                    self.category_popularity[cat_slug] = products_in_cat[:100]
        print(f"  Category popularity: {len(self.category_popularity)} categories")
    
    def get_global_recommendations(self, exclude: Set[int] = None, n: int = 6) -> List[int]:
        exclude = exclude or set()
        return [p for p in self.global_popularity if p not in exclude][:n]
    
    def get_category_recommendations(self, category_slugs: List[str], exclude: Set[int] = None, n: int = 6) -> List[int]:
        exclude = exclude or set()
        recommendations = []
        for cat_slug in category_slugs:
            if cat_slug in self.category_popularity:
                for prod in self.category_popularity[cat_slug]:
                    if prod not in exclude and prod not in recommendations:
                        recommendations.append(prod)
                        if len(recommendations) >= n:
                            return recommendations
        return recommendations

class CooccurrenceModel:
    """Product co-occurrence from session co-views and orders."""
    def __init__(self):
        self.cooccurrence = defaultdict(Counter)
        self.copurchase = defaultdict(Counter)
    
    def fit_from_sessions(self, hits: pd.DataFrame, weight: float = 1.0):
        """Learn co-occurrence from session co-views."""
        print("Fitting co-occurrence from sessions...")
        product_hits = hits[hits['page_type'] == 'PRODUCT'].copy()
        session_products = product_hits.groupby('visit_id')['product_id'].apply(
            lambda x: list(x.dropna().unique()))
        pair_count = 0
        for products in tqdm(session_products, desc="  Processing sessions"):
            if len(products) >= 2:
                for i, p1 in enumerate(products):
                    for p2 in products[i+1:]:
                        p1, p2 = int(p1), int(p2)
                        self.cooccurrence[p1][p2] += weight
                        self.cooccurrence[p2][p1] += weight
                        pair_count += 1
        print(f"  Processed {pair_count:,} co-occurrence pairs")
    
    def fit_from_orders(self, orders: pd.DataFrame, weight: float = 2.0):
        """Learn co-purchase patterns from order data."""
        print("Fitting co-purchase from orders...")
        order_products = orders.groupby('id')['product_id'].apply(list)
        pair_count = 0
        for products in tqdm(order_products, desc="  Processing orders"):
            if len(products) >= 2:
                for i, p1 in enumerate(products):
                    for p2 in products[i+1:]:
                        p1, p2 = int(p1), int(p2)
                        self.copurchase[p1][p2] += weight
                        self.copurchase[p2][p1] += weight
                        pair_count += 1
        print(f"  Processed {pair_count:,} co-purchase pairs")
    
    def merge_old_site_data(self, old_cooccurrence: 'CooccurrenceModel', 
                           old_to_new_mapping: Dict[int, int], discount: float = 0.5):
        """Merge old site co-occurrence data (mapped to new IDs)."""
        print(f"Merging old site data (discount={discount})...")
        merged_count = 0
        for old_p1, counter in old_cooccurrence.cooccurrence.items():
            new_p1 = old_to_new_mapping.get(old_p1)
            if new_p1:
                for old_p2, count in counter.items():
                    new_p2 = old_to_new_mapping.get(old_p2)
                    if new_p2:
                        self.cooccurrence[new_p1][new_p2] += count * discount
                        merged_count += 1
        print(f"  Merged {merged_count:,} pairs from old site")
    
    def get_recommendations(self, viewed_products: List[int], exclude: Set[int] = None, n: int = 6) -> List[int]:
        if not viewed_products: return []
        exclude = exclude or set()
        scores = Counter()
        for product in viewed_products:
            product = int(product)
            for co_prod, count in self.cooccurrence[product].items():
                if co_prod not in exclude:
                    scores[co_prod] += count
            for co_prod, count in self.copurchase[product].items():
                if co_prod not in exclude:
                    scores[co_prod] += count
        return [p for p, _ in scores.most_common(n)]
    
    def get_item_recommendations(self, item: int, exclude: Set[int] = None, n: int = 6) -> List[int]:
        """Get recommendations for a single item (used for time-weighted scoring)."""
        exclude = exclude or set()
        item = int(item)
        scores = Counter()
        for co_prod, count in self.cooccurrence[item].items():
            if co_prod not in exclude:
                scores[co_prod] += count
        for co_prod, count in self.copurchase[item].items():
            if co_prod not in exclude:
                scores[co_prod] += count
        return [p for p, _ in scores.most_common(n)]

class SequentialTransitionModel:
    """Model for sequential A→B transition patterns."""
    def __init__(self):
        self.immediate_transitions = defaultdict(Counter)
        self.skip_transitions = defaultdict(Counter)
    
    def fit(self, hits: pd.DataFrame):
        """Learn transition patterns from ordered product sequences."""
        print("Fitting sequential transition model...")
        product_hits = hits[hits['page_type'] == 'PRODUCT'].copy()
        product_hits = product_hits.sort_values(['visit_id', 'date_time'])
        session_sequences = product_hits.groupby('visit_id')['product_id'].apply(
            lambda x: [int(p) for p in x.dropna().tolist()])
        transition_count = 0
        for sequence in tqdm(session_sequences, desc="  Processing sequences"):
            if len(sequence) < 2: continue
            for i in range(len(sequence) - 1):
                self.immediate_transitions[sequence[i]][sequence[i+1]] += 1
                transition_count += 1
            for i in range(len(sequence) - 2):
                self.skip_transitions[sequence[i]][sequence[i+2]] += 0.5
        print(f"  Processed {transition_count:,} transitions")
    
    def get_recommendations(self, viewed_products: List[int], exclude: Set[int] = None, 
                          n: int = 6, last_item_boost: float = 2.0) -> List[int]:
        if not viewed_products: return []
        exclude = exclude or set()
        scores = Counter()
        n_viewed = len(viewed_products)
        for i, product in enumerate(viewed_products):
            product = int(product)
            recency_weight = (i + 1) / n_viewed
            for next_prod, count in self.immediate_transitions[product].items():
                if next_prod not in exclude:
                    scores[next_prod] += recency_weight * count
            for next_prod, count in self.skip_transitions[product].items():
                if next_prod not in exclude:
                    scores[next_prod] += recency_weight * 0.5 * count
        if viewed_products:
            last_product = int(viewed_products[-1])
            for next_prod, count in self.immediate_transitions[last_product].items():
                if next_prod not in exclude:
                    scores[next_prod] += last_item_boost * count
        return [p for p, _ in scores.most_common(n)]

class SearchQueryMatcher:
    """Match search queries to products for cold-start recommendations."""
    def __init__(self, products_df: pd.DataFrame):
        self.products = products_df[['id', 'name', 'brand', 'main_category']].copy()
        self.products['name_lower'] = self.products['name'].fillna('').str.lower()
        self.products['brand_lower'] = self.products['brand'].fillna('').str.lower()
    
    def match_query(self, query: str, n: int = 10) -> List[int]:
        query_lower = query.lower().strip()
        if not query_lower: return []
        scores = []
        query_words = query_lower.split()
        for _, product in self.products.iterrows():
            score = 0
            name = product['name_lower']
            brand = product['brand_lower']
            if query_lower in brand or brand in query_lower: score += 10
            for word in query_words:
                if len(word) > 2 and word in name: score += 5
            if query_lower in name: score += 15
            if score > 0: scores.append((product['id'], score))
        scores.sort(key=lambda x: -x[1])
        return [pid for pid, _ in scores[:n]]

class SemanticModel:
    """Semantic similarity model using pre-trained product embeddings.
    
    Uses embeddings trained via Contrastive Learning (Skip-gram) to find
    semantically similar products based on cosine similarity.
    """
    
    def __init__(self, embeddings_path: str = None, product_ids_path: str = None):
        self.embeddings = None
        self.product_map = {}
        self.index_to_pid = {}
        self.enabled = False
        
        if embeddings_path and product_ids_path:
            try:
                self._load_embeddings(embeddings_path, product_ids_path)
            except Exception as e:
                print(f"  Warning: Could not load semantic embeddings: {e}")
    
    def _load_embeddings(self, embeddings_path: str, product_ids_path: str):
        """Load pre-trained embeddings from files."""
        import os
        if not os.path.exists(embeddings_path) or not os.path.exists(product_ids_path):
            print(f"  Semantic model files not found, skipping...")
            return
        
        print(f"  Loading semantic embeddings from {embeddings_path}...")
        self.embeddings = np.load(embeddings_path)
        
        with open(product_ids_path, 'r') as f:
            product_ids = json.load(f)
        
        self.product_map = {int(pid): i for i, pid in enumerate(product_ids)}
        self.index_to_pid = {i: int(pid) for pid, i in self.product_map.items()}
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
        self.embeddings = self.embeddings / norms
        
        self.enabled = True
        print(f"  ✓ Loaded {len(self.product_map):,} product embeddings (dim={self.embeddings.shape[1]})")
    
    def get_recommendations(self, viewed_products: List[int], 
                           exclude: Set[int] = None, n: int = 6) -> List[int]:
        """Get recommendations based on embedding similarity.
        
        Uses weighted average of all viewed products' embeddings (with recency weighting).
        """
        if not self.enabled or not viewed_products:
            return []
        
        exclude = exclude or set()
        
        # Collect embeddings of all viewed products with recency weighting
        valid_embeddings = []
        weights = []
        n_products = len(viewed_products)
        
        for i, item in enumerate(viewed_products):
            item = int(item)
            if item in self.product_map:
                valid_embeddings.append(self.embeddings[self.product_map[item]])
                # Recency weight: later items get higher weight
                weights.append((i + 1) / n_products)
        
        if not valid_embeddings:
            return []
        
        # Compute weighted average embedding
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        query_vec = np.average(valid_embeddings, axis=0, weights=weights)
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        
        # Compute cosine similarity with all products
        scores = np.dot(self.embeddings, query_vec)
        
        # Get top-N excluding viewed products
        top_indices = np.argsort(scores)[::-1]
        recommendations = []
        exclude_set = set(int(p) for p in viewed_products) | exclude
        
        for idx in top_indices:
            pid = self.index_to_pid[idx]
            if pid not in exclude_set and pid not in recommendations:
                recommendations.append(pid)
                if len(recommendations) >= n:
                    break
        
        return recommendations

class GRU4RecModel:
    """GRU4Rec neural model for session-based recommendations.
    
    Loads pre-trained PyTorch model and provides recommendations
    based on learned sequential patterns.
    """
    
    def __init__(self, model_path: str = None):
        self.enabled = False
        self.model = None
        self.product_to_idx = {}
        self.idx_to_product = {}
        
        # Try to load model
        if model_path is None:
            # Auto-detect paths
            paths_to_try = [
                'models/gru4rec_best.pth',
                '/kaggle/input/gru4rec-model/gru4rec_best.pth',
            ]
            for path in paths_to_try:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained GRU4Rec model."""
        try:
            import torch
            import torch.nn as nn
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            self.product_to_idx = checkpoint['product_to_idx']
            self.idx_to_product = checkpoint['idx_to_product']
            cfg = checkpoint['config']
            
            # Define model architecture
            class GRU4Rec(nn.Module):
                def __init__(self, num_items, embedding_dim, hidden_dim, num_layers=1, dropout=0.2):
                    super().__init__()
                    self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
                    self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, 
                                      batch_first=True, dropout=dropout if num_layers > 1 else 0)
                    self.dropout = nn.Dropout(dropout)
                    self.output = nn.Linear(hidden_dim, num_items)
                
                def forward(self, x, lengths):
                    embedded = self.embedding(x)
                    embedded = self.dropout(embedded)
                    packed = nn.utils.rnn.pack_padded_sequence(
                        embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
                    _, hidden = self.gru(packed)
                    out = self.dropout(hidden[-1])
                    return self.output(out)
            
            # Create and load model
            self.model = GRU4Rec(
                num_items=cfg['num_items'],
                embedding_dim=cfg['embedding_dim'],
                hidden_dim=cfg['hidden_dim'],
                num_layers=cfg['num_layers'],
                dropout=cfg['dropout']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.enabled = True
            print(f"  ✓ Loaded GRU4Rec model ({cfg['num_items']} products)")
            
        except Exception as e:
            print(f"  ✗ Failed to load GRU4Rec: {e}")
            self.enabled = False
    
    def get_recommendations(self, viewed_products: List[int], 
                           exclude: Set[int] = None, n: int = 6) -> List[int]:
        """Get recommendations from GRU4Rec model."""
        if not self.enabled or not viewed_products:
            return []
        
        import torch
        
        exclude = exclude or set()
        
        # Convert product IDs to indices
        indices = []
        for pid in viewed_products:
            pid = int(pid)
            if pid in self.product_to_idx:
                indices.append(self.product_to_idx[pid])
        
        if not indices:
            return []
        
        # Prepare input tensor
        input_seq = torch.LongTensor([indices])
        lengths = torch.LongTensor([len(indices)])
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(input_seq, lengths)
        
        # Get top-k predictions
        _, top_k = torch.topk(logits[0], n * 3)  # Get more to account for exclusions
        
        # Convert back to product IDs
        recommendations = []
        exclude_set = set(int(p) for p in viewed_products) | exclude
        
        for idx in top_k.tolist():
            if idx in self.idx_to_product:
                pid = self.idx_to_product[idx]
                if pid not in exclude_set and pid not in recommendations:
                    recommendations.append(pid)
                    if len(recommendations) >= n:
                        break
        
        return recommendations

class EnsembleRecommender:
    """Combines multiple recommendation signals with smart fallback cascade.
    
    Tier hierarchy:
    0. GRU4Rec neural model (learned sequential patterns)
    1. Sequential transitions (A→B patterns from last items)
    2. Co-occurrence (items viewed together in sessions)
    2.5. Semantic similarity (embedding-based)
    3. Same-category boost (NEW)
    4. Category popularity
    5. Search query matching
    6. Global popularity fallback
    """
    def __init__(self, popularity_model, cooccurrence_model, transition_model, 
                 search_matcher, config, mappings=None, semantic_model=None, gru4rec_model=None):
        self.popularity = popularity_model
        self.cooccurrence = cooccurrence_model
        self.transitions = transition_model
        self.search_matcher = search_matcher
        self.config = config
        self.semantic = semantic_model
        self.gru4rec = gru4rec_model
        self.mappings = mappings or {}
    
    def predict(self, session_features: Dict, n: int = 6) -> List[int]:
        viewed_products = session_features.get('product_ids', [])
        viewed_categories = session_features.get('category_slugs', [])
        search_queries = session_features.get('search_queries', [])
        exclude = set(viewed_products)
        
        # Collect candidates with scores from multiple sources
        candidate_scores = Counter()
        
        # SOURCE 1: Sequential transitions (weight: 3.0)
        if len(viewed_products) >= 1:
            trans_recs = self.transitions.get_recommendations(
                viewed_products, exclude=exclude, n=30, last_item_boost=self.config.LAST_ITEM_BOOST)
            for i, r in enumerate(trans_recs):
                candidate_scores[r] += 3.0 * (1.0 - i * 0.03)
        
        # SOURCE 2: Co-occurrence (weight: 2.0)
        if viewed_products:
            cooccur_recs = self.cooccurrence.get_recommendations(
                viewed_products, exclude=exclude, n=30)
            for i, r in enumerate(cooccur_recs):
                candidate_scores[r] += 2.0 * (1.0 - i * 0.03)
        
        # SOURCE 3: Semantic similarity (weight: 2.5)
        if self.semantic and self.semantic.enabled and viewed_products:
            semantic_recs = self.semantic.get_recommendations(
                viewed_products, exclude=exclude, n=30)
            for i, r in enumerate(semantic_recs):
                candidate_scores[r] += 2.5 * (1.0 - i * 0.03)
        
        # SOURCE 3.5: ORDER-BASED BOOST (weight: 2.0) - Products actually purchased
        order_counts = self.mappings.get('order_counts', {})
        if order_counts:
            for pid, cnt in order_counts.items():
                if pid not in exclude and pid in candidate_scores:
                    candidate_scores[pid] += 2.0 * min(cnt / 10.0, 1.0)
        
        # SOURCE 4: Category popularity (weight: 0.5)
        if viewed_categories:
            cat_recs = self.popularity.get_category_recommendations(
                viewed_categories, exclude=exclude, n=20)
            for i, r in enumerate(cat_recs):
                candidate_scores[r] += 0.5 * (1.0 - i * 0.05)
        
        
        # Get top N by combined score
        recommendations = [p for p, _ in candidate_scores.most_common(n)]
        
        # FALLBACK: Search query matching
        if search_queries and len(recommendations) < n:
            for query in search_queries:
                search_recs = self.search_matcher.match_query(query, n=n)
                for r in search_recs:
                    if r not in recommendations and r not in exclude:
                        recommendations.append(r)
                        if len(recommendations) >= n: break
                if len(recommendations) >= n: break
        
        # FALLBACK: Global popularity
        if len(recommendations) < n:
            global_recs = self.popularity.get_global_recommendations(
                exclude=exclude | set(recommendations), n=n)
            for r in global_recs:
                if r not in recommendations:
                    recommendations.append(r)
                    if len(recommendations) >= n: break
        
        return recommendations[:n]

# ==============================
# 9. TRAINING PIPELINE
# ==============================
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# Separate new/old site data
new_site_visits = visits[visits['project_id'] == 0]
old_site_visits = visits[visits['project_id'] == 1]
new_site_hits = hits[hits['project_id'] == 0]
old_site_hits = hits[hits['project_id'] == 1]

# RECENCY FILTER: Only use data from 2025-07+ (matching test period - BEST)
RECENCY_CUTOFF = '2025-07-01'
new_site_visits_recent = new_site_visits[new_site_visits['date_time'] >= RECENCY_CUTOFF]
print(f"\nFiltering to recent data (>= {RECENCY_CUTOFF})...")
print(f"  New site visits: {len(new_site_visits):,} -> {len(new_site_visits_recent):,} ({len(new_site_visits_recent)/len(new_site_visits)*100:.1f}%)")

# Get recent watch_ids for filtering hits
recent_visit_ids = set(new_site_visits_recent['visit_id'].unique())
# Build watch_ids from recent visits
recent_watch_ids = set()
for _, row in new_site_visits_recent.iterrows():
    try:
        watch_ids = ast.literal_eval(str(row['watch_ids']))
        recent_watch_ids.update([int(w) for w in watch_ids])
    except:
        pass
new_site_hits_recent = new_site_hits[new_site_hits['watch_id'].isin(recent_watch_ids)]
print(f"  New site hits: {len(new_site_hits):,} -> {len(new_site_hits_recent):,}")

# Use recent data for training
new_site_visits = new_site_visits_recent
new_site_hits = new_site_hits_recent

print(f"\nNew site (RECENT): {len(new_site_visits):,} visits, {len(new_site_hits):,} hits")
print(f"Old site: {len(old_site_visits):,} visits, {len(old_site_hits):,} hits")

# Build session mappings
print("\nBuilding session mappings...")
new_watch_to_visit = build_watch_to_visit_mapping(new_site_visits)
old_watch_to_visit = build_watch_to_visit_mapping(old_site_visits)

# Prepare hits data
print("\nPreparing hits data...")
new_site_hits = prepare_hits_data(new_site_hits, new_watch_to_visit, mappings['new_slug_to_id'])
old_site_hits = prepare_hits_data(old_site_hits, old_watch_to_visit, mappings['old_slug_to_id'],
                                 is_old_site=True, old_slug_to_new_id=mappings['old_slug_to_new_id'])

# Train Popularity Model
popularity_model = PopularityModel()
popularity_model.fit(new_site_hits, CATEGORY_SLUG_TO_RUSSIAN, category_to_products)

# Train Co-occurrence Model
cooccurrence_model = CooccurrenceModel()
cooccurrence_model.fit_from_sessions(new_site_hits, weight=config.COOCCURRENCE_WEIGHT)
cooccurrence_model.fit_from_orders(new_orders, weight=config.COPURCHASE_WEIGHT)

# Add old site co-occurrence
old_cooccurrence = CooccurrenceModel()
old_cooccurrence.fit_from_sessions(old_site_hits, weight=1.0)
cooccurrence_model.merge_old_site_data(old_cooccurrence, mappings['old_to_new_id'], discount=config.OLD_SITE_DISCOUNT)

# Add old site orders
old_orders_mapped = old_orders.copy()
old_orders_mapped['product_id'] = old_orders_mapped['product_id'].map(mappings['old_to_new_id'])
old_orders_mapped = old_orders_mapped.dropna(subset=['product_id'])
old_orders_mapped['product_id'] = old_orders_mapped['product_id'].astype(int)
if len(old_orders_mapped) > 0:
    cooccurrence_model.fit_from_orders(old_orders_mapped, weight=config.COPURCHASE_WEIGHT * config.OLD_SITE_DISCOUNT)

# Train Transition Model
transition_model = SequentialTransitionModel()
transition_model.fit(new_site_hits)

# Build cross-category transition patterns from sessions
print("Building cross-category patterns...")
cat_transitions = defaultdict(Counter)
cat_product_popularity = defaultdict(lambda: defaultdict(Counter))
product_to_category = mappings.get('product_to_category', {})

# Group hits by visit to get sessions
visit_products = defaultdict(list)
for _, row in new_site_hits.iterrows():
    if pd.notna(row.get('product_id')):
        visit_products[row['visit_id']].append(int(row['product_id']))

for visit_id, products in visit_products.items():
    if len(products) >= 2:
        for i in range(len(products) - 1):
            p1, p2 = products[i], products[i+1]
            cat1 = product_to_category.get(p1)
            cat2 = product_to_category.get(p2)
            if cat1 and cat2 and cat1 != cat2:
                cat_transitions[cat1][cat2] += 1
                cat_product_popularity[cat1][cat2][p2] += 1

mappings['cat_transitions'] = dict(cat_transitions)
mappings['cat_product_popularity'] = {k: dict(v) for k, v in cat_product_popularity.items()}
print(f"  Cross-category pairs: {sum(len(v) for v in cat_transitions.values())}")

# Create Search Matcher
search_matcher = SearchQueryMatcher(new_products)

# Create Semantic Model (553 products for better coverage)
print("\nInitializing semantic model...")
EMBEDDINGS_PATH = 'embeddings/item_embeddings.npy'
PRODUCT_IDS_PATH = 'embeddings/product_ids.json'
# On Kaggle, check alternative paths
import os
if os.path.exists('/kaggle/input/rental-embeddings/item_embeddings.npy'):
    EMBEDDINGS_PATH = '/kaggle/input/rental-embeddings/item_embeddings.npy'
    PRODUCT_IDS_PATH = '/kaggle/input/rental-embeddings/product_ids.json'
semantic_model = SemanticModel(EMBEDDINGS_PATH, PRODUCT_IDS_PATH)

# Initialize GRU4Rec model
print("\nInitializing GRU4Rec model...")
gru4rec_model = GRU4RecModel()

# Create Ensemble
ensemble = EnsembleRecommender(
    popularity_model, cooccurrence_model, transition_model, 
    search_matcher, config, mappings=mappings, semantic_model=semantic_model, gru4rec_model=gru4rec_model
)
print("\n✓ All models trained successfully!")

# ==============================
# 10. GENERATE SUBMISSION
# ==============================
def generate_submission(ensemble, test_visits, test_hits, mappings, output_path='submission.csv'):
    """Generate submission file."""
    print("\n" + "="*60)
    print("GENERATING SUBMISSION")
    print("="*60)
    
    print("\nBuilding test session mappings...")
    test_watch_to_visit = build_watch_to_visit_mapping(test_visits)
    
    print("Preparing test hits...")
    test_hits_prepared = prepare_hits_data(test_hits, test_watch_to_visit, mappings['new_slug_to_id'])
    
    test_visit_ids = test_visits['visit_id'].tolist()
    print(f"\nProcessing {len(test_visit_ids):,} test sessions...")
    session_features = extract_all_session_features(test_hits_prepared, test_visit_ids)
    
    print("\nGenerating predictions...")
    predictions = []
    
    for visit_id in tqdm(test_visit_ids, desc="Predicting"):
        features = session_features.get(visit_id, {})
        recs = ensemble.predict(features, n=config.NUM_PREDICTIONS)
        
        # Validate predictions
        valid_recs = []
        seen = set()
        for r in recs:
            if r in mappings['valid_new_ids'] and r not in seen:
                valid_recs.append(r)
                seen.add(r)
        
        # Fill with popular if needed
        if len(valid_recs) < config.NUM_PREDICTIONS:
            for p in ensemble.popularity.global_popularity:
                if p not in seen and p in mappings['valid_new_ids']:
                    valid_recs.append(p)
                    seen.add(p)
                    if len(valid_recs) >= config.NUM_PREDICTIONS:
                        break
        
        predictions.append({
            'visit_id': visit_id,
            'product_ids': ' '.join(map(str, valid_recs[:config.NUM_PREDICTIONS]))
        })
    
    submission = pd.DataFrame(predictions)
    
    print("\nValidating submission...")
    assert len(submission) == len(test_visits), "Missing predictions!"
    for idx, row in submission.iterrows():
        products = row['product_ids'].split()
        assert len(products) == 6, f"Row {idx} has {len(products)} products"
        assert len(set(products)) == 6, f"Row {idx} has duplicates"
    print("✓ Validation passed!")
    
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to {output_path}")
    print(f"  Total rows: {len(submission):,}")
    return submission

# Generate final submission
submission = generate_submission(ensemble, visits_test, hits_test, mappings)

# ==============================
# 11. ANALYSIS
# ==============================
print("\n" + "="*60)
print("PREDICTION ANALYSIS")
print("="*60)

all_recs = []
for _, row in submission.iterrows():
    all_recs.extend(row['product_ids'].split())
rec_counts = Counter(all_recs)
print(f"\nUnique products recommended: {len(rec_counts)}")
print(f"Total recommendations: {len(all_recs):,}")

print("\nMost frequently recommended products:")
for prod_id, count in rec_counts.most_common(10):
    prod_id = int(prod_id)
    slug = mappings['new_id_to_slug'].get(prod_id, 'Unknown')
    print(f"  {prod_id}: {count:,} times ({slug[:50]}...)")

print("\n" + "="*60)
print("NOTEBOOK COMPLETE!")
print("="*60)

# %% [markdown]
# # Conclusion:
# 
# The baseline model provides a simple but valid starting point, giving predictions based on the most frequently viewed products.
# 
# Data exploration revealed that certain products and categories dominate user sessions, which can be leveraged for recommendations.
# 
# For improved performance, sequential models (RNN/GRU/Transformer) and feature engineering at both product and session levels are recommended.
# 
# Proper mapping of session identifiers (watch_id) and product slugs to IDs ensures the pipeline works correctly for prediction and submission.