# CL-GRU4Rec+RP Web Demo

Interactive web demo for CL-GRU4Rec+RP recommendation system with **real data** and **FastAPI backend**.

## Tech Stack

- **Next.js 16** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS v4** - Styling
- **FastAPI** - Python backend with real model
- **Mock Auth** - localStorage-based authentication

## Features

- 🔐 **Mock Authentication**: Sign in with any email/password
- 🛒 **Real Product Catalog**: 689 products from Kaggle Rental Product dataset
- 🎯 **Real-time Recommendations**: Updates as you add products to session
- 📊 **Component Scores**: See GRU4Rec, Contrastive Learning, and Re-Purchase scores
- 💡 **Session Tracking**: Visualize current session and remove items
- 📁 **Category Filtering**: Filter products by 27+ Russian product categories

## Getting Started

### 1. Start FastAPI Backend

```bash
cd backend
source ../../venv/bin/activate  # or your Python venv
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

API will be available at http://localhost:8001

### 2. Start Next.js Frontend

```bash
npm run dev --turbo
```

Frontend will be available at http://localhost:3000

## Demo Credentials

Any email/password works! Quick test:

- Email: `demo@example.com`
- Password: `demo123`

Or use pre-configured:

- Email: `user@demo.com`
- Password: any

## How to Use

1. **Login**: Enter any email/password
2. **Browse Products**: View catalog on the left (689 real Russian rental products)
3. **Add to Session**: Click any product to add to your session
4. **View Recommendations**: Watch real-time recommendations on the right
5. **See Component Scores**: Each recommendation shows:
   - **GRU**: Sequential pattern score
   - **CL**: Contrastive learning similarity
   - **RP**: Re-purchase frequency
6. **Continue Shopping**: Click recommended products to extend session
7. **Remove Items**: Use X button to remove from session

## API Endpoints

- `GET /health` - Health check
- `GET /api/products?limit=100` - Get products
- `GET /api/products/{id}` - Get single product
- `GET /api/categories` - Get all categories
- `POST /api/recommend` - Get recommendations
- `GET /api/popular?limit=50` - Get popular products

## Component Architecture

### FastAPI Backend (`/backend`)

- **data_loader.py**: Loads real data from `/Users/macbook/Desktop/product-recommendation-system/data/`
  - 689 products from `new_site_products.csv`
  - Session data from `metrika_hits.csv`
  - 27+ product categories in Russian

- **model_wrapper.py**: Wraps CL-GRU4Rec+RP model
  - Uses `GRU4RecModel` from parent directory
  - Falls back to popularity-based recommendations when no checkpoint
  - To use trained model: Add checkpoint path to `.env`

- **main.py**: FastAPI application
  - CORS enabled for localhost:3000
  - RESTful API for products and recommendations

### Next.js Frontend

- `src/app/page.tsx`: Main demo page
- `src/components/`: ProductCard, RecommendationPanel, SessionTracker
- `src/lib/api-client.ts`: Fetches from FastAPI backend
- `src/contexts/AuthContext.tsx`: Mock authentication

## Notes

- **Real Data**: Products are from Russian rental platform (Synerise RecSys 2025)
- **Model**: Currently using popularity-based fallback (no trained checkpoint loaded)
- **To use trained model**: Train the model first, then add checkpoint path to backend

## Future Enhancements

- Train and load actual CL-GRU4Rec+RP checkpoint
- Add user history persistence
- Include evaluation metrics dashboard
- Add more product filters (price range, brand)
