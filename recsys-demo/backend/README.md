# CL-GRU4Rec+RP FastAPI Backend

Real recommendation API using the actual PyTorch model.

## Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Run the server:**
```bash
python -m app.main
```

Or with uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Access API:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /api/products` - Get all products (with optional `?category=` and `?limit=` params)
- `GET /api/products/{id}` - Get single product
- `GET /api/categories` - Get all categories
- `POST /api/recommend` - Get recommendations for a session
- `GET /api/popular?limit=50` - Get popular products

## Model

The backend uses the real CL-GRU4Rec+RP model from:
`/Users/macbook/Desktop/product-recommendation-system/cl_gru4rec_rp_unified.py`

## Data

Real data is loaded from:
- Products: `/Users/macbook/Desktop/product-recommendation-system/data/new_site_products.csv`
- Sessions: `/Users/macbook/Desktop/product-recommendation-system/data/metrika_hits.csv`
