from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os

import database
from database import SessionLocal, engine

database.init_db()
database.populate_sample_data()

app = FastAPI(
    title="Noon Analytics API",
    description="API for scraping and analyzing Noon.com product data."
)

@app.get("/", include_in_schema=False)
async def read_root():
    return FileResponse('frontend/index.html')

@app.get("/dashboard", include_in_schema=False)
async def dashboard():
    return FileResponse('frontend/dashboard.html')

@app.get("/static/{path:path}", include_in_schema=False)
async def serve_static(path: str):
    static_path = os.path.join('frontend', path)
    if os.path.exists(static_path):
        return FileResponse(static_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/scrape", status_code=202)
async def api_scrape(payload: dict):
    url = payload.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required.")
    # In a real application, this would trigger a background task (e.g., Celery)
    return {"message": "Scraping task initiated.", "task_id": f"task_{hash(url)}"}

@app.get("/api/products", summary="Get Filtered Products")
async def get_products(
    db: Session = Depends(database.get_db),
    search: Optional[str] = Query(None, description="Search term for product name"),
    category: Optional[List[str]] = Query(None, description="List of categories to filter by"),
    brand: Optional[str] = Query(None, description="Brand to filter by (searches in product name)"),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price")
):
    products = database.get_all_products(
        db=db,
        search=search,
        categories=category,
        brand=brand,
        min_price=min_price,
        max_price=max_price
    )
    return products

@app.get("/api/filter-options", summary="Get Filter Options")
async def get_filters(db: Session = Depends(database.get_db)):
    options = database.get_filter_options(db)
    return options 