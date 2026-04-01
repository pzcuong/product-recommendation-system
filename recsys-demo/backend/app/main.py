"""
CL-GRU4Rec+RP FastAPI Backend
Real recommendation API with actual PyTorch model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

from .data_loader import data_loader
from .model_wrapper import get_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CL-GRU4Rec+RP API",
    description="Contrastive Learning Enhanced GRU4Rec with Re-Purchase Awareness",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class RecommendationRequest(BaseModel):
    session_items: List[str]
    k: Optional[int] = 10
    client_id: Optional[str] = None


class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    confidence: float
    session_length: int
    component_weights: dict


class Product(BaseModel):
    id: str
    name: str
    brand: Optional[str] = ""
    main_category: str
    categories: str
    price: float
    description: Optional[str] = ""
    slug: str


class ProductsResponse(BaseModel):
    products: List[Product]
    total: int
    categories: List[str]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting CL-GRU4Rec+RP API...")
    model = get_model()
    if model._initialized:
        logger.info("Model initialized successfully")
    else:
        logger.warning("Model failed to initialize - using fallback")


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CL-GRU4Rec+RP Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "products": "/api/products",
            "recommend": "/api/recommend",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    model = get_model()
    return {
        "status": "healthy",
        "model_loaded": model._initialized if model else False,
        "products_loaded": len(data_loader.product_map) > 0
    }


@app.get("/api/products", response_model=ProductsResponse)
async def get_products(
    category: Optional[str] = None,
    limit: Optional[int] = None
):
    """Get all products or filter by category"""
    try:
        if category:
            products = data_loader.get_products_by_category(category)
        else:
            products = data_loader.get_all_products()

        # Get popular products if limit specified
        if limit and not category:
            popular_ids = data_loader.get_popular_products(limit)
            product_map = {p['id']: p for p in products}
            products = [
                product_map.get(pid, product_map[pid])
                for pid in popular_ids if pid in product_map
            ]

        return ProductsResponse(
            products=[
                Product(
                    id=p['id'],
                    name=p['name'],
                    brand=p.get('brand', ''),
                    main_category=p.get('main_category', ''),
                    categories=p.get('categories', ''),
                    price=p.get('price', 0),
                    description=p.get('description', '')[:200] if p.get('description') else '',
                    slug=p.get('slug', '')
                )
                for p in products[:limit] if limit else products
            ],
            total=len(products),
            categories=data_loader.get_categories()
        )
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/products/{product_id}", response_model=Product)
async def get_product(product_id: str):
    """Get a single product by ID"""
    product = data_loader.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    return Product(**product)


@app.get("/api/categories")
async def get_categories():
    """Get all product categories"""
    categories = data_loader.get_categories()
    return {"categories": categories}


@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Generate recommendations for a session"""
    try:
        model = get_model()

        if not model._initialized:
            # Fallback: return popular products
            logger.warning("Model not initialized, using fallback")
            popular_ids = data_loader.get_popular_products(request.k or 10)

            recommendations = [
                {
                    "product_id": pid,
                    "score": 1.0 - (i * 0.05),
                    "gru_score": 0.5,
                    "cl_score": 0.3,
                    "rp_score": 0.2
                }
                for i, pid in enumerate(popular_ids[:request.k or 10])
            ]

            return RecommendationResponse(
                recommendations=recommendations,
                confidence=0.5,
                session_length=len(request.session_items),
                component_weights={"gru": 0.5, "cl": 0.3, "rp": 0.2}
            )

        # Get real recommendations
        recs = model.predict(request.session_items, k=request.k or 10)

        # Calculate confidence
        confidence = model.get_confidence(len(request.session_items))

        # Calculate component weights based on session length
        session_len = len(request.session_items)
        if session_len < 3:
            weights = {"gru": 0.5, "cl": 0.25, "rp": 0.25}
        elif session_len < 7:
            weights = {"gru": 0.65, "cl": 0.15, "rp": 0.20}
        else:
            weights = {"gru": 0.80, "cl": 0.10, "rp": 0.10}

        return RecommendationResponse(
            recommendations=recs,
            confidence=confidence,
            session_length=session_len,
            component_weights=weights
        )

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/popular")
async def get_popular_products(limit: int = 50):
    """Get popular products for cold start"""
    try:
        popular_ids = data_loader.get_popular_products(limit)
        products = []

        for pid in popular_ids:
            product = data_loader.get_product(pid)
            if product:
                products.append(Product(**product))

        return {"products": products[:limit]}

    except Exception as e:
        logger.error(f"Error getting popular products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
