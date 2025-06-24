import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./products.db")

MAX_PRODUCTS_PER_SCRAPE = int(os.getenv("MAX_PRODUCTS_PER_SCRAPE", "100"))
SCRAPING_TIMEOUT = int(os.getenv("SCRAPING_TIMEOUT", "120"))
PAGE_LOAD_DELAY = int(os.getenv("PAGE_LOAD_DELAY", "3"))

CLASSIFICATION_ENABLED = os.getenv("CLASSIFICATION_ENABLED", "true").lower() == "true"
CLASSIFICATION_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.7"))
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B/32")

NOON_BASE_URL = "https://www.noon.com"
NOON_UAE_BASE = "https://www.noon.com/uae-en" 