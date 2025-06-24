from celery import Celery
from database import SessionLocal
from scraper import scrape_products
from classifier import classify_product
from config import REDIS_URL, MAX_PRODUCTS_PER_SCRAPE, CLASSIFICATION_ENABLED

celery_app = Celery(
    "noon_scraper",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"]
)

@celery_app.task(bind=True)
def process_scrape_request(self, url: str):
    db = SessionLocal()
    try:
        raw_products = scrape_products(url, max_products=MAX_PRODUCTS_PER_SCRAPE)
        for product_data in raw_products:
            if CLASSIFICATION_ENABLED and product_data.get("image_url"):
                classification_result = classify_product(
                    image_url=product_data["image_url"],
                    product_name=product_data.get("name", "")
                )
                product_data.update(classification_result)
            # Save product_data to database (not implemented here)
    finally:
        db.close() 