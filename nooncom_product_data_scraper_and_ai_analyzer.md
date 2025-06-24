# **Noon.com Scraper & Analyzer: Project Documentation**

### **Introduction**

This document provides a complete guide to the Noon.com Scraper & Analyzer project. The application is designed to scrape product data from the Noon.com e-commerce platform, perform AI-driven classification on the scraped items, and present the analyzed data in a dynamic, interactive web dashboard.

The architecture is built on a modern Python stack, featuring:
- **FastAPI**: For the high-performance web server and API.
- **Celery & Redis**: For managing asynchronous background tasks (scraping and classification).
- **Playwright & BeautifulSoup**: For robust web scraping.
- **PyTorch & CLIP**: For AI-powered image-based product classification.
- **SQLAlchemy**: For database interaction, with SQLite as the default.
- **Vanilla JS & Tailwind CSS**: For a responsive and interactive frontend dashboard.

This guide contains everything required to set up the environment, run the application, and understand the source code.

---

## **1. Setup and Execution Guide**

This section provides step-by-step instructions for setting up the environment and running the application on a macOS system. For other operating systems, please refer to the `README.md` file for specific commands (e.g., virtual environment activation).

### **1.1. Prerequisite Tool Installation**

These are one-time installations of foundational software.

1.  **Install Homebrew (Package Manager for macOS)**
    Open your terminal and execute the following command to install Homebrew, which simplifies software installation.
    ```shell
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```

2.  **Install Python**
    Use Homebrew to install a modern version of Python.
    ```shell
    brew install python
    ```

3.  **Install Redis**
    Use Homebrew to install Redis, which acts as the message broker for the background task queue.
    ```shell
    brew install redis
    ```

### **1.2. Project Environment Configuration**

These steps prepare the project's source code and its specific dependencies.

1.  **Download the Project Code**
    Clone the project repository. *Note: Replace `<your-repository-url>` with the actual Git URL.*
    ```shell
    git clone <your-repository-url>
    cd noon-scraper-analyzer
    ```

2.  **Create and Activate a Virtual Environment**
    This creates an isolated environment for the project's Python packages.
    ```shell
    # Create the virtual environment
    python3 -m venv venv

    # Activate the environment
    source venv/bin/activate
    ```
    > **Important:** You must activate the virtual environment (`source venv/bin/activate`) every time you open a new terminal window to work on this project.

3.  **Install Required Python Packages**
    Install all Python libraries listed in the `requirements.txt` file.
    ```shell
    pip install -r requirements.txt
    ```

4.  **Install Browser Engines for Playwright**
    Download the headless browser binaries that the web scraper will control.
    ```shell
    playwright install
    ```

5.  **Initialize the Database**
    Run the database script to create the necessary tables and populate them with sample data.
    ```shell
    python database.py
    ```

### **1.3. Running the Application**

The application consists of three services that must be run concurrently in **three separate terminal windows**.

#### **Terminal 1: Start the Redis Server**

1.  Open a new terminal window.
2.  Run the command to start the Redis server. It must remain running in the background.
    ```shell
    redis-server
    ```

#### **Terminal 2: Start the Celery Worker**

1.  Open a second terminal window.
2.  Navigate to the project directory and activate the virtual environment.
3.  Run the command to start the background task worker. This process handles all scraping and data processing jobs.
    ```shell
    celery -A tasks.celery_app worker --loglevel=info
    ```
    > *Note for Windows users:* Use `celery -A tasks.celery_app worker --loglevel=info --pool=solo` for compatibility.

#### **Terminal 3: Start the FastAPI Web Server**

1.  Open a third terminal window.
2.  Navigate to the project directory and activate the virtual environment.
3.  Run the command to start the web application server.
    ```shell
    uvicorn main:app --reload
    ```

### **1.4. Access and Use the Application**

1.  **Access the App**: Once all three services are running, open your web browser and navigate to: **`http://127.0.0.1:8000`**
2.  **Start Scraping**: On the landing page, paste a Noon.com category or search URL (e.g., `https://www.noon.com/uae-en/fashion/women`) and click "Start Analysis".
3.  **View Dashboard**: You will be automatically redirected to the dashboard page (`dashboard.html`), where results will appear as they are processed. The data will refresh automatically.

To shut down the application, press `Ctrl + C` in each of the three terminal windows.

---

## **2. Project Files and Source Code**

This section contains the complete directory structure and source code for every file in the project.

### **2.1. Project Directory Structure**

```
noon-scraper-analyzer/
├── data/
│   └── products.json
├── frontend/
│   ├── dashboard.css
│   ├── dashboard.html
│   ├── dashboard.js
│   └── index.html
├── .env
├── README.md
├── classifier.py
├── config.py
├── database.py
├── main.py
├── requirements.txt
├── scraper.py
└── tasks.py
```

### **2.2. Source Code**

<br>

**File: `README.md`**
```markdown
# **Noon.com Scraper & Analyzer**

This web application scrapes, classifies, and analyzes product data from Noon.com. It features an asynchronous task-based architecture with a FastAPI backend and a dynamic frontend dashboard.

### **Prerequisites**

Before you begin, ensure you have the following installed on your system:
*   **Python** (version 3.9 or newer)
*   **Git**
*   **Redis**: [Installation Guide](https://redis.io/docs/getting-started/installation/)

### **Step 1: Clone the Repository**

First, clone the project repository to your local machine.

```shell
git clone <your-repository-url>
cd noon-scraper-analyzer
```

### **Step 2: Set Up Python Environment**

It is highly recommended to use a virtual environment to manage project dependencies.

```shell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### **Step 3: Install Dependencies**

Install all required Python packages using the `requirements.txt` file.

```shell
pip install -r requirements.txt
```

Next, install the necessary browser binaries for Playwright.

```shell
playwright install
```

### **Step 4: Configure Environment Variables**

The application uses a `.env` file for configuration. Create a file named `.env` in the project root directory and add the following variables.

```env
# URL for your Redis instance
REDIS_URL="redis://localhost:6379/0"

# Database connection string. SQLite is used by default.
# For PostgreSQL, use: "postgresql://user:password@host:port/dbname"
DATABASE_URL="sqlite:///./products.db"

# Set to "false" to disable AI classification and speed up scraping
CLASSIFICATION_ENABLED="true"

# Maximum number of products to scrape in a single run
MAX_PRODUCTS_PER_SCRAPE="100"
```

### **Step 5: Initialize the Database**

Run the database script to create the necessary tables and populate them with some initial sample data.

```shell
python database.py
```

### **Step 6: Running the Application**

The application consists of three main components that need to be run in separate terminal windows: **Redis**, the **Celery Worker**, and the **FastAPI Server**.

**Terminal 1: Start Redis**
If Redis is not already running as a service, start it manually:
```shell
redis-server
```

**Terminal 2: Start the Celery Worker**
With your virtual environment activated, start the Celery worker. This process will wait for and execute background jobs.
```shell
celery -A tasks.celery_app worker --loglevel=info --pool=solo
```
*Note: The `--pool=solo` flag is recommended for Windows compatibility.*

**Terminal 3: Start the FastAPI Server**
Finally, start the main web server using Uvicorn.
```shell
uvicorn main:app --reload
```

### **Step 7: Using the Application**

1.  **Open the App**: Open your web browser and navigate to `http://127.0.0.1:8000`. You will see the URL submission page.
2.  **Start Scraping**: Paste a valid Noon.com category or search URL (e.g., `https://www.noon.com/uae-en/fashion/women`) and click **"Start Analysis"**.
3.  **Monitor Progress**: You will be redirected to the dashboard. The scraping and classification will happen in the background. You can monitor the progress in the terminal where the Celery worker is running.
4.  **Analyze Data**: As products are processed and saved, they will appear on the dashboard. Use the filters on the left sidebar to search, filter by category/price, and sort the results. The charts and product grid will update in real-time based on your selections.
```

---
**File: `.env`**
```env
# URL for your Redis instance
REDIS_URL="redis://localhost:6379/0"

# Database connection string. SQLite is used by default.
# For PostgreSQL, use: "postgresql://user:password@host:port/dbname"
DATABASE_URL="sqlite:///./products.db"

# Set to "false" to disable AI classification and speed up scraping
CLASSIFICATION_ENABLED="true"

# Maximum number of products to scrape in a single run
MAX_PRODUCTS_PER_SCRAPE="100"
```

---
**File: `requirements.txt`**
```txt
fastapi==0.104.1
uvicorn==0.24.0
celery==5.3.4
redis==5.0.1
sqlalchemy==2.0.23
pandas==2.1.3
playwright==1.40.0
beautifulsoup4==4.12.2
openai-clip==1.0.1
torch==2.1.1
torchvision==0.16.1
Pillow==10.1.0
openpyxl==3.1.2
requests==2.31.0
aiohttp==3.9.1
numpy==1.24.3
python-multipart==0.0.6
python-dotenv==1.0.0
pydantic==2.5.0
transformers==4.35.2
pytest==7.4.3
```

---
**File: `config.py`**
```python
import os
from typing import List, Dict, Any

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

PRODUCT_URL_PATTERN = r'/uae-en/[^/]+/([A-Z0-9]{20,25})/p/'

SCRAPING_SELECTORS = {
    "product_container": "div[data-qa='product-tile']",
    "product_link": "a[href*='/p/']",
    "product_name": "[data-qa='product-name']",
    "product_price": "[data-qa='product-price']",
    "product_discount": "[data-qa='product-discount']",
    "product_image": "img[data-qa='product-image']",
    "delivery_info": "[data-qa='delivery-info']"
}

CLASSIFICATION_CATEGORIES: Dict[str, List[str]] = {
    "main_categories": [
        "a photo of clothing",
        "a photo of shoes", 
        "a photo of a bag",
        "a photo of accessories",
        "a photo of electronics",
        "a photo of home decor",
        "a photo of furniture"
    ],
    "colors": [
        "black", "white", "red", "blue", "green", "yellow", "orange", "purple",
        "pink", "brown", "gray", "navy", "beige", "gold", "silver", "multicolor", "khaki"
    ],
    "materials": [
        "cotton", "leather", "synthetic", "wool", "silk", "denim", "polyester",
        "canvas", "suede", "metal", "plastic", "fabric", "mesh", "linen", "viscose", "aluminum"
    ],
    "patterns": [
        "solid", "striped", "floral", "geometric", "abstract", "animal print",
        "polka dots", "checkered", "paisley", "embroidered", "plain", "textured"
    ],
    "occasions": [
        "casual", "formal", "sport", "party", "work", "outdoor", "beach",
        "wedding", "travel", "everyday", "special occasion"
    ],
    "garment_types": [
        "shirt", "dress", "pants", "skirt", "jacket", "sweater", "t-shirt",
        "blouse", "jeans", "shorts", "coat", "hoodie", "top", "bottom", "peacoat"
    ],
    "shoe_types": [
        "sneakers", "heels", "boots", "sandals", "flats", "loafers", "pumps",
        "athletic shoes", "dress shoes", "casual shoes", "slippers"
    ],
    "bag_types": [
        "handbag", "backpack", "shoulder bag", "tote bag", "clutch", "crossbody bag", "messenger bag",
        "duffel bag", "wallet", "purse", "briefcase", "travel bag"
    ]
}

BROWSER_CONFIG = {
    "headless": True,
    "timeout": SCRAPING_TIMEOUT * 1000,
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "viewport": {"width": 1920, "height": 1080},
    "ignore_https_errors": True
}
```

---
**File: `database.py`**
```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, distinct
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import os
from typing import List, Optional

from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False, index=True)
    product_link = Column(String, nullable=False)
    price = Column(Float, nullable=True)
    currency = Column(String, default="AED")
    discount_percentage = Column(Integer, nullable=True)
    delivery_type = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    category = Column(String, nullable=True, index=True)
    colour = Column(String, nullable=True)
    material = Column(String, nullable=True)
    pattern = Column(String, nullable=True)
    occasion = Column(String, nullable=True)
    garment_type = Column(String, nullable=True)
    shoe_type = Column(String, nullable=True)
    bag_type = Column(String, nullable=True)
    scraped_at = Column(DateTime, default=datetime.utcnow, index=True)
    classification_confidence = Column(Float, nullable=True)
    raw_data = Column(Text, nullable=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
    
def get_all_products(db: Session, search: Optional[str] = None, categories: Optional[List[str]] = None, brand: Optional[str] = None, min_price: Optional[float] = None, max_price: Optional[float] = None):
    query = db.query(Product)
    
    if search:
        query = query.filter(Product.name.ilike(f'%{search}%'))
        
    if categories:
        query = query.filter(Product.category.in_(categories))
        
    if brand:
        query = query.filter(Product.name.ilike(f'%{brand}%'))
        
    if min_price is not None:
        query = query.filter(Product.price >= min_price)
        
    if max_price is not None:
        query = query.filter(Product.price <= max_price)
        
    return query.order_by(Product.scraped_at.desc()).all()

def get_filter_options(db: Session):
    categories_query = db.query(distinct(Product.category)).filter(Product.category != None, Product.category != '').all()
    categories = sorted([c[0] for c in categories_query])
    
    return {
        "categories": categories
    }

def populate_sample_data():
    db = SessionLocal()
    try:
        if db.query(Product).count() > 0:
            print("Database already contains data. Skipping sample data population.")
            return
            
        print("Populating database with sample data...")
        sample_products = [
            Product(product_id='N40893810A', name='Classic Crew Neck T-Shirt', product_link='https://www.noon.com/uae-en/p/', price=79.0, currency='AED', discount_percentage=20, delivery_type='Express', image_url='https://images.pexels.com/photos/428338/pexels-photo-428338.jpeg?auto=compress&cs=tinysrgb&w=600', category='Clothing', colour='White', material='Cotton', scraped_at=datetime.fromisoformat('2025-06-22T22:10:00Z')),
            Product(product_id='Z1E2844DB0', name='June Embroidered Loose Fit Shirt', product_link='https://www.noon.com/uae-en/p/', price=189.0, currency='AED', discount_percentage=None, delivery_type='Market', image_url='https://images.pexels.com/photos/769749/pexels-photo-769749.jpeg?auto=compress&cs=tinysrgb&w=600', category='Clothing', colour='Blue', material='Linen', scraped_at=datetime.fromisoformat('2025-06-22T22:15:00Z')),
            Product(product_id='N53346754A', name='Air Zoom Pegasus Running Shoes', product_link='https://www.noon.com/uae-en/p/', price=450.0, currency='AED', discount_percentage=15, delivery_type='Express', image_url='https://images.pexels.com/photos/1032110/pexels-photo-1032110.jpeg?auto=compress&cs=tinysrgb&w=600', category='Shoes', colour='Black', material='Mesh', scraped_at=datetime.fromisoformat('2025-06-22T21:05:00Z')),
            Product(product_id='N12345678B', name='Leather Crossbody Bag', product_link='https://www.noon.com/uae-en/p/', price=320.5, currency='AED', discount_percentage=None, delivery_type='Express', image_url='https://images.pexels.com/photos/1152077/pexels-photo-1152077.jpeg?auto=compress&cs=tinysrgb&w=600', category='Bags', colour='Brown', material='Leather', scraped_at=datetime.fromisoformat('2025-06-21T18:30:00Z')),
            Product(product_id='N98765432C', name='Wireless Noise Cancelling Headphones', product_link='https://www.noon.com/uae-en/p/', price=899.0, currency='AED', discount_percentage=10, delivery_type='Express', image_url='https://images.pexels.com/photos/3945683/pexels-photo-3945683.jpeg?auto=compress&cs=tinysrgb&w=600', category='Electronics', colour='Silver', material='Plastic', scraped_at=datetime.fromisoformat('2025-06-22T20:00:00Z')),
            Product(product_id='N24681357D', name='Slim Fit Denim Jeans', product_link='https://www.noon.com/uae-en/p/', price=250.0, currency='AED', discount_percentage=30, delivery_type='Market', image_url='https://images.pexels.com/photos/52573/jeans-pants-blue-shop-52573.jpeg?auto=compress&cs=tinysrgb&w=600', category='Clothing', colour='Blue', material='Denim', scraped_at=datetime.fromisoformat('2025-06-22T19:45:00Z')),
        ]
        
        db.add_all(sample_products)
        db.commit()
        print(f"Added {len(sample_products)} sample products to the database.")
        
    except Exception as e:
        print(f"Error populating sample data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database schema...")
    init_db()
    populate_sample_data()
    print("Database initialization complete.")
```

---
**File: `main.py`**
```python
from fastapi import FastAPI, Depends, HTTPException, Query, BackgroundTasks, FileResponse
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
from typing import List, Optional
import os

import database
from tasks import process_scrape_request
from database import get_db, Product

# Initialize the database and create tables
database.init_db()

app = FastAPI(
    title="Noon.com Scraper & Analyzer",
    version="2.0",
    description="An API to scrape product data from Noon.com, classify it using AI, and serve it to an analytics dashboard."
)

class ScrapeRequest(BaseModel):
    url: HttpUrl

@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse('frontend/index.html')

@app.get("/{file_path:path}", include_in_schema=False)
async def serve_static_files(file_path: str):
    # This serves frontend files like dashboard.html, css, and js
    safe_path = os.path.join("frontend", file_path)
    if os.path.exists(safe_path):
        return FileResponse(safe_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/api/scrape", status_code=202)
async def create_scrape_task(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Accepts a Noon.com URL and initiates a background task to scrape and analyze it.
    """
    task = process_scrape_request.delay(str(request.url))
    return {"message": "Scraping task initiated.", "task_id": task.id}

@app.get("/api/products", response_model=List[Product], summary="Get Filtered Products")
async def get_products(
    db: Session = Depends(get_db),
    search: Optional[str] = Query(None, description="Search term for product name"),
    category: Optional[List[str]] = Query(None, description="List of categories to filter by"),
    brand: Optional[str] = Query(None, description="Brand to filter by (searches in product name)"),
    min_price: Optional[float] = Query(None, gte=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, gte=0, description="Maximum price")
):
    """
    Retrieves a list of products from the database, with optional server-side filtering.
    """
    products = database.get_all_products(
        db=db,
        search=search,
        categories=category,
        brand=brand,
        min_price=min_price,
        max_price=max_price
    )
    return products
    
@app.get("/api/filter-options", summary="Get Dynamic Filter Options")
async def get_filters(db: Session = Depends(get_db)):
    """
    Retrieves available dynamic filter options, such as all unique product categories currently in the database.
    """
    options = database.get_filter_options(db)
    return options

# This is a workaround for the FastAPI/Uvicorn startup to populate sample data
# It runs after the app starts up.
@app.on_event("startup")
def on_startup():
    database.populate_sample_data()
```

---
**File: `scraper.py`**
```python
from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional
from config import (
    NOON_BASE_URL, 
    PRODUCT_URL_PATTERN, 
    SCRAPING_SELECTORS,
    BROWSER_CONFIG,
    PAGE_LOAD_DELAY
)

logger = logging.getLogger(__name__)

class NoonProductScraper:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
        self.product_pattern = re.compile(PRODUCT_URL_PATTERN)
        
    async def initialize_browser(self):
        """Initialize Playwright browser instance"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=BROWSER_CONFIG["headless"],
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            logger.info("Browser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise
    
    async def get_page(self) -> Page:
        """Get a new browser page."""
        if not self.browser:
            await self.initialize_browser()
        context = await self.browser.new_context(**BROWSER_CONFIG)
        page = await context.new_page()
        await page.set_default_timeout(BROWSER_CONFIG["timeout"])
        return page

    async def navigate_and_scroll(self, page: Page, url: str) -> str:
        """Navigate to URL and scroll to load all products"""
        try:
            logger.info(f"Navigating to: {url}")
            await page.goto(url, wait_until="networkidle")
            await asyncio.sleep(PAGE_LOAD_DELAY)
            
            # Scroll down to trigger dynamic content loading
            last_height = await page.evaluate("document.body.scrollHeight")
            for _ in range(5): # Limit scrolls to prevent infinite loops
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(2) # Wait for new content to load
                new_height = await page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            content = await page.content()
            logger.info("Page content loaded successfully")
            return content
        except Exception as e:
            logger.error(f"Error navigating and scrolling page {url}: {e}")
            raise

    def extract_product_data(self, html_content: str, max_products: int) -> List[Dict[str, Any]]:
        """Extract product data from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        products = []
        seen_ids = set()

        product_containers = soup.select(SCRAPING_SELECTORS["product_container"])
        logger.info(f"Found {len(product_containers)} product containers")
        
        for container in product_containers:
            if len(products) >= max_products:
                break
            try:
                product_data = self._extract_single_product(container)
                if product_data and product_data["product_id"] not in seen_ids:
                    products.append(product_data)
                    seen_ids.add(product_data["product_id"])
            except Exception as e:
                logger.warning(f"Could not parse a product container: {e}")
        
        logger.info(f"Successfully extracted {len(products)} unique products")
        return products
    
    def _extract_single_product(self, container) -> Optional[Dict[str, Any]]:
        """Extract data from a single product container"""
        link_element = container.select_one(SCRAPING_SELECTORS["product_link"])
        if not link_element or not link_element.get('href'):
            return None

        href = link_element['href']
        if not href.startswith('http'):
            href = NOON_BASE_URL + href
        
        match = self.product_pattern.search(href)
        if not match:
            return None
        
        product_id = match.group(1)
        
        name = (container.select_one(SCRAPING_SELECTORS["product_name"]) or {}).get_text(strip=True)
        price_text = (container.select_one(SCRAPING_SELECTORS["product_price"]) or {}).get_text(strip=True)
        discount_text = (container.select_one(SCRAPING_SELECTORS["product_discount"]) or {}).get_text(strip=True)
        image_url = (container.select_one(SCRAPING_SELECTORS["product_image"]) or {}).get('src')
        delivery_type = (container.select_one(SCRAPING_SELECTORS["delivery_info"]) or {}).get_text(strip=True)

        if not all([product_id, name, price_text, image_url]):
            return None

        return {
            "product_id": product_id,
            "name": name,
            "product_link": href,
            "price": self._clean_price(price_text),
            "currency": "AED",
            "discount_percentage": self._clean_discount(discount_text),
            "delivery_type": delivery_type,
            "image_url": image_url,
            "raw_data": str(container)[:500]
        }
    
    def _clean_price(self, price_text: str) -> Optional[float]:
        match = re.search(r'(\d[\d,]*\.?\d*)', price_text or "")
        return float(match.group(1).replace(',', '')) if match else None

    def _clean_discount(self, discount_text: str) -> Optional[int]:
        match = re.search(r'(\d+)%', discount_text or "")
        return int(match.group(1)) if match else None

    async def scrape_products(self, url: str, max_products: int) -> List[Dict[str, Any]]:
        """Main scraping method orchestrating the process."""
        page = await self.get_page()
        try:
            html_content = await self.navigate_and_scroll(page, url)
            return self.extract_product_data(html_content, max_products)
        finally:
            await page.close()
    
    async def cleanup(self):
        """Clean up browser resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Scraper resources cleaned up.")
```

---
**File: `classifier.py`**
```python
import clip
import torch
from PIL import Image
import requests
from typing import Dict, Any, Optional, List
import logging
import io
from config import (
    CLIP_MODEL_NAME, 
    CLASSIFICATION_CATEGORIES,
    CLASSIFICATION_CONFIDENCE_THRESHOLD
)

logger = logging.getLogger(__name__)

class ProductClassifier:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProductClassifier, cls).__new__(cls)
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.model, cls._instance.preprocess = None, None
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """Load CLIP model and preprocessing pipeline"""
        if self.model is None:
            try:
                logger.info(f"Loading CLIP model '{CLIP_MODEL_NAME}' to {self.device}")
                self.model, self.preprocess = clip.load(CLIP_MODEL_NAME, device=self.device)
                logger.info("CLIP model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load CLIP model: {e}")
                raise

    def _download_image(self, image_url: str) -> Optional[Image.Image]:
        try:
            response = requests.get(image_url, timeout=10, stream=True)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return image
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not download image from {image_url}: {e}")
        return None

    def _classify_attribute(self, image_features: torch.Tensor, text_prompts: List[str]) -> Dict[str, Any]:
        text_inputs = clip.tokenize(text_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            
            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            best_idx = similarity.argmax().item()
            confidence = similarity[0, best_idx].item()
            prediction_raw = text_prompts[best_idx]
            
            # Clean up the prediction label
            prediction = prediction_raw.replace("a photo of ", "").replace("a photo of a ", "")

        return {"prediction": prediction, "confidence": confidence}

    def classify_product(self, image_url: str, product_name: str = "") -> Optional[Dict[str, Any]]:
        """Main classification method for a single product image."""
        logger.info(f"Classifying product: {product_name} from {image_url}")
        image = self._download_image(image_url)
        if not image:
            return None

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

        classified_attributes = {}
        confidences = []

        # Hierarchical classification: first main category, then specific attributes
        category_result = self._classify_attribute(image_features, CLASSIFICATION_CATEGORIES["main_categories"])
        if category_result['confidence'] >= CLASSIFICATION_CONFIDENCE_THRESHOLD:
            main_category = category_result['prediction']
            classified_attributes["category"] = main_category
            confidences.append(category_result['confidence'])
            
            # Now classify sub-attributes based on the main category
            specific_key = f"{main_category.lower()}_types"
            if specific_key in CLASSIFICATION_CATEGORIES:
                specific_result = self._classify_attribute(image_features, CLASSIFICATION_CATEGORIES[specific_key])
                if specific_result['confidence'] >= CLASSIFICATION_CONFIDENCE_THRESHOLD:
                    classified_attributes[f"{main_category.lower()}_type"] = specific_result['prediction']

        # Classify general attributes
        for attr, prompts in [("colour", "colors"), ("material", "materials"), ("pattern", "patterns"), ("occasion", "occasions")]:
            result = self._classify_attribute(image_features, CLASSIFICATION_CATEGORIES[prompts])
            if result['confidence'] >= CLASSIFICATION_CONFIDENCE_THRESHOLD:
                classified_attributes[attr] = result['prediction']
                confidences.append(result['confidence'])

        if confidences:
            classified_attributes["classification_confidence"] = sum(confidences) / len(confidences)
        
        logger.info(f"Classification result: {classified_attributes}")
        return classified_attributes
```

---
**File: `tasks.py`**
```python
from celery import Celery
from datetime import datetime
import asyncio
import logging
from typing import Dict, Any

from database import SessionLocal, Product
from scraper import NoonProductScraper
from classifier import ProductClassifier
# data_processor isn't a real file in the provided list, so I'm removing it. 
# Based on the code, the functionality is handled within the task itself.
from config import REDIS_URL, MAX_PRODUCTS_PER_SCRAPE, CLASSIFICATION_ENABLED

# Setup Celery
celery_app = Celery("noon_scraper_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="tasks.process_scrape_request")
def process_scrape_request(self, url: str) -> Dict[str, Any]:
    """
    Celery task to scrape a URL, classify products, and save to the database.
    """
    logger.info(f"Task {self.request.id}: Starting for URL {url}")
    self.update_state(state='PROGRESS', meta={'status': 'Initializing...'})
    
    scraper = NoonProductScraper()
    classifier = ProductClassifier() if CLASSIFICATION_ENABLED else None
    
    db = SessionLocal()
    processed_count = 0
    
    try:
        # Run async scraping code
        self.update_state(state='PROGRESS', meta={'status': 'Scraping products...'})
        loop = asyncio.get_event_loop()
        raw_products = loop.run_until_complete(scraper.scrape_products(url, MAX_PRODUCTS_PER_SCRAPE))
        
        total_products = len(raw_products)
        logger.info(f"Task {self.request.id}: Scraped {total_products} raw products.")
        
        for i, cleaned_data in enumerate(raw_products):
            status_meta = {
                'status': f'Processing {i+1}/{total_products}',
                'product_name': cleaned_data.get('name', 'N/A')
            }
            self.update_state(state='PROGRESS', meta=status_meta)

            if not cleaned_data or not cleaned_data.get('product_id'):
                continue
            
            # AI Classification
            if classifier and cleaned_data.get('image_url'):
                status_meta['status'] = f'Classifying {i+1}/{total_products}'
                self.update_state(state='PROGRESS', meta=status_meta)
                
                class_attrs = classifier.classify_product(cleaned_data['image_url'], cleaned_data['name'])
                if class_attrs:
                    cleaned_data.update(class_attrs)
            
            # Database operation
            existing_product = db.query(Product).filter(Product.product_id == cleaned_data["product_id"]).first()
            if existing_product:
                for key, value in cleaned_data.items():
                    setattr(existing_product, key, value)
                existing_product.scraped_at = datetime.utcnow()
            else:
                new_product = Product(**cleaned_data)
                db.add(new_product)
            
            processed_count += 1

        db.commit()
        logger.info(f"Task {self.request.id}: Committed {processed_count} products to the database.")

    except Exception as e:
        logger.error(f"Task {self.request.id}: Failed with error: {e}", exc_info=True)
        db.rollback()
        self.update_state(state='FAILURE', meta={'status': 'Failed', 'error': str(e)})
        # Re-raise the exception so Celery knows the task failed
        raise
    finally:
        db.close()
        # Clean up async resources
        loop.run_until_complete(scraper.cleanup())

    return {'status': 'Completed', 'processed_products': processed_count, 'url': url}
```

---
**File: `frontend/index.html`**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Noon Analytics - Start</title>
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
    </style>
</head>
<body class="bg-gray-900 text-white flex items-center justify-center min-h-screen">

    <div class="w-full max-w-2xl text-center p-8">
        <header class="mb-12">
            <h1 class="text-5xl font-bold text-gray-100 mb-2">Analyze Noon.com Products Instantly</h1>
            <p class="text-lg text-gray-400">Paste a Noon.com category or search URL to begin.</p>
        </header>

        <main class="flex flex-col sm:flex-row items-center gap-4">
            <div class="relative flex-grow w-full">
                <input 
                    type="url" 
                    id="noon-url-input"
                    placeholder="Paste a Noon.com URL here"
                    class="w-full h-14 px-6 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none transition duration-300 text-gray-200 placeholder-gray-500"
                >
            </div>
            <button 
                id="start-analysis-btn"
                class="w-full sm:w-auto h-14 px-8 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-500 transition-all duration-300 transform hover:scale-105 flex items-center justify-center gap-2"
                onclick="startAnalysis()">
                <span>Start Analysis</span>
                <svg id="arrow-icon" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-arrow-right"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
            </button>
        </main>

        <footer class="mt-24 text-gray-600 text-sm">
            <p>Clicking "Start Analysis" will trigger a background job to scrape and analyze the URL. You will be redirected to the dashboard to see the results.</p>
        </footer>
    </div>

    <script>
        async function startAnalysis() {
            const button = document.getElementById('start-analysis-btn');
            const input = document.getElementById('noon-url-input');
            const url = input.value;

            if (!url || !url.startsWith('https://www.noon.com')) {
                alert('Please paste a valid Noon.com URL.');
                return;
            }
            
            button.disabled = true;
            const originalButtonContent = button.innerHTML;
            button.innerHTML = `
                <svg class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Initiating...</span>
            `;

            try {
                const response = await fetch('/api/scrape', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url }),
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ detail: 'An unknown server error occurred.' }));
                    throw new Error(errorData.detail);
                }
                
                const result = await response.json();
                
                // Redirect to the dashboard immediately after queueing the task
                window.location.href = 'dashboard.html';

            } catch (error) {
                console.error('Analysis initiation failed:', error);
                alert(`Error: ${error.message}`);
                button.disabled = false;
                button.innerHTML = originalButtonContent;
            }
        }
    </script>
</body>
</html>
```

---
**File: `frontend/dashboard.html`**
```html
<!DOCTYPE html>
<html lang="en" class="bg-gray-900">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Noon Analytics - Dashboard</title>
    <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="dashboard.css">
</head>
<body class="bg-gray-900 text-gray-200 font-sans">
    <div id="app-container" class="flex min-h-screen">
        <!-- Sidebar -->
        <aside class="w-72 bg-gray-900/70 backdrop-blur-sm border-r border-gray-800 p-6 flex flex-col fixed h-full">
            <header class="flex items-center gap-3 mb-8">
                 <div class="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center">
                    <i data-lucide="bar-chart-3" class="text-white"></i>
                 </div>
                <h1 class="text-2xl font-bold text-white">Noon<span class="text-indigo-400">Lytics</span></h1>
            </header>

            <div class="space-y-6 flex-grow overflow-y-auto pr-2 custom-scrollbar">
                <!-- Search Filter -->
                <div>
                    <label for="search" class="text-sm font-medium text-gray-400 block mb-2">Search by Name</label>
                    <div class="relative">
                        <i data-lucide="search" class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500"></i>
                        <input type="text" id="search" placeholder="Search products..." class="w-full bg-gray-800 border border-gray-700 rounded-md pl-10 pr-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>
                </div>

                <!-- Category Filter -->
                <div>
                    <label class="text-sm font-medium text-gray-400 block mb-3">Category</label>
                    <div id="category-filters" class="space-y-2">
                        <!-- Checkboxes will be injected here -->
                    </div>
                </div>
                
                <!-- Price Range Filter -->
                <div>
                    <label class="text-sm font-medium text-gray-400 block mb-2">Price Range (AED)</label>
                    <div class="flex items-center gap-2">
                        <input type="number" id="price-min" placeholder="Min" class="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-center focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                        <span class="text-gray-500">-</span>
                        <input type="number" id="price-max" placeholder="Max" class="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-center focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                    </div>
                </div>

                 <!-- Sort Options -->
                <div>
                    <label for="sort-by" class="text-sm font-medium text-gray-400 block mb-2">Sort By</label>
                    <select id="sort-by" class="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                        <option value="scraped_at_desc">Date Scraped (Newest)</option>
                        <option value="price_asc">Price (Low to High)</option>
                        <option value="price_desc">Price (High to Low)</option>
                    </select>
                </div>
            </div>

            <footer class="mt-6 pt-6 border-t border-gray-800 text-center">
                <p class="text-xs text-gray-500">&copy; 2025 NoonLytics. All rights reserved.</p>
            </footer>
        </aside>

        <!-- Main Content -->
        <main class="ml-72 flex-1 p-8 bg-gray-900">
            <div class="max-w-7xl mx-auto">
                <header class="flex justify-between items-center mb-6">
                    <div>
                        <h2 class="text-3xl font-bold text-white">Analytics Dashboard</h2>
                        <p class="text-gray-400">Showing <span id="product-count">0</span> products.</p>
                    </div>
                </header>
                
                <!-- Visualizations -->
                <section class="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-6">
                    <div class="lg:col-span-2 bg-gray-800/50 p-6 rounded-xl border border-gray-700/50">
                        <h3 class="font-semibold text-white mb-4">Products by Category</h3>
                        <div class="chart-container h-64">
                            <canvas id="category-chart"></canvas>
                        </div>
                    </div>
                    <div class="lg:col-span-3 bg-gray-800/50 p-6 rounded-xl border border-gray-700/50">
                        <h3 class="font-semibold text-white mb-4">Price Distribution (AED)</h3>
                        <div class="chart-container h-64">
                            <canvas id="price-chart"></canvas>
                        </div>
                    </div>
                </section>

                <!-- Product Grid -->
                <section id="product-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                    <!-- Product cards will be injected here -->
                </section>
                <div id="no-results" class="hidden text-center py-20">
                    <i data-lucide="search-x" class="mx-auto h-16 w-16 text-gray-500"></i>
                    <h3 class="mt-4 text-xl font-semibold text-white">No Products Found</h3>
                    <p class="mt-1 text-gray-400">Try adjusting your filters or wait for the scrape to complete.</p>
                </div>
            </div>
        </main>
    </div>

    <script type="module" src="dashboard.js"></script>
</body>
</html>
```

---
**File: `frontend/dashboard.css`**
```css
body {
    font-family: 'Inter', sans-serif;
    background-color: #111827; /* bg-gray-900 */
}

/* Custom Scrollbar for Sidebar */
.custom-scrollbar::-webkit-scrollbar {
    width: 6px;
}

.custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
}

.custom-scrollbar::-webkit-scrollbar-thumb {
    background: #374151; /* bg-gray-700 */
    border-radius: 3px;
}

.custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background: #4B5563; /* bg-gray-600 */
}

/* To make Chart.js responsive within its container */
.chart-container {
    position: relative;
    width: 100%;
}

/* Remove number input spinners */
input[type='number']::-webkit-inner-spin-button,
input[type='number']::-webkit-outer-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

input[type='number'] {
  -moz-appearance: textfield;
}

/* Custom checkbox styling */
.custom-checkbox {
    appearance: none;
    background-color: #374151; /* bg-gray-700 */
    border: 1px solid #4B5563; /* border-gray-600 */
    border-radius: 4px;
    width: 1.25rem;
    height: 1.25rem;
    cursor: pointer;
    display: inline-block;
    position: relative;
    top: 0.25rem;
    transition: background-color 0.2s, border-color 0.2s;
}

.custom-checkbox:checked {
    background-color: #6366F1; /* bg-indigo-500 */
    border-color: #6366F1;
}

.custom-checkbox:checked::after {
    content: '';
    position: absolute;
    left: 6px;
    top: 2px;
    width: 5px;
    height: 10px;
    border: solid white;
    border-width: 0 2px 2px 0;
    transform: rotate(45deg);
}
```

---
**File: `frontend/dashboard.js`**
```javascript
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();

    const state = {
        products: [],
        filters: {
            search: '',
            categories: new Set(),
            priceMin: null,
            priceMax: null,
        },
        sortBy: 'scraped_at_desc',
    };

    let categoryChart, priceChart;
    let filterTimeout;

    const elements = {
        grid: document.getElementById('product-grid'),
        count: document.getElementById('product-count'),
        noResults: document.getElementById('no-results'),
        categoryFilters: document.getElementById('category-filters'),
        searchInput: document.getElementById('search'),
        priceMinInput: document.getElementById('price-min'),
        priceMaxInput: document.getElementById('price-max'),
        sortBySelect: document.getElementById('sort-by'),
    };

    async function init() {
        try {
            setupCharts();
            await setupFilters();
            await fetchAndRenderProducts(); // Initial fetch
            addEventListeners();
            // Poll for new data every 5 seconds
            setInterval(fetchAndRenderProducts, 5000);
        } catch (error) {
            console.error("Failed to initialize dashboard:", error);
            elements.grid.innerHTML = `<p class="text-red-400">Error loading data from the server. Please try again later.</p>`;
        }
    }

    async function setupFilters() {
        try {
            const response = await fetch('/api/filter-options');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            const categories = data.categories || [];
            
            elements.categoryFilters.innerHTML = categories.map(cat => `
                <div class="flex items-center">
                    <input type="checkbox" id="cat-${cat.toLowerCase().replace(/\s+/g, '-')}" data-category="${cat}" class="custom-checkbox mr-2">
                    <label for="cat-${cat.toLowerCase().replace(/\s+/g, '-')}" class="text-gray-300 cursor-pointer select-none">${cat}</label>
                </div>
            `).join('');
        } catch (error) {
            console.error("Failed to setup filters:", error);
            elements.categoryFilters.innerHTML = `<p class="text-red-400 text-xs">Could not load categories.</p>`;
        }
    }

    async function fetchAndRenderProducts() {
        const params = new URLSearchParams();
        if (state.filters.search) params.append('search', state.filters.search);
        state.filters.categories.forEach(cat => params.append('category', cat));
        if (state.filters.priceMin !== null) params.append('min_price', state.filters.priceMin);
        if (state.filters.priceMax !== null) params.append('max_price', state.filters.priceMax);

        try {
            const response = await fetch(`/api/products?${params.toString()}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            state.products = await response.json();
            
            sortProducts();
            
            renderProductGrid(state.products);
            updateCharts(state.products);
        } catch (error) {
            console.error("Failed to fetch products:", error);
        }
    }
    
    function sortProducts() {
        const [sortKey, sortOrder] = state.sortBy.split('_');
        state.products.sort((a, b) => {
            const valA = sortKey === 'scraped_at' ? new Date(a[sortKey]) : a[sortKey];
            const valB = sortKey === 'scraped_at' ? new Date(b[sortKey]) : b[sortKey];
            if (valA < valB) return sortOrder === 'asc' ? -1 : 1;
            if (valA > valB) return sortOrder === 'asc' ? 1 : -1;
            return 0;
        });
    }
    
    function setupCharts() {
        const chartOptions = (isCategoryChart) => ({
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: '#9CA3AF' } },
                x: { grid: { display: false }, ticks: { color: '#9CA3AF' } }
            }
        });
        categoryChart = new Chart(document.getElementById('category-chart'), { type: 'bar', data: {}, options: chartOptions(true) });
        priceChart = new Chart(document.getElementById('price-chart'), { type: 'bar', data: {}, options: chartOptions(false) });
    }

    function updateCharts(products) {
        // Category Chart
        const categoryCounts = products.reduce((acc, p) => {
            if (p.category) acc[p.category] = (acc[p.category] || 0) + 1;
            return acc;
        }, {});
        const sortedCategories = Object.entries(categoryCounts).sort((a, b) => b[1] - a[1]);
        categoryChart.data = {
            labels: sortedCategories.map(c => c[0]),
            datasets: [{ label: 'Products', data: sortedCategories.map(c => c[1]), backgroundColor: '#6366F1', borderRadius: 4 }]
        };
        categoryChart.update();

        // Price Chart
        const prices = products.map(p => p.price).filter(p => p !== null);
        const maxPrice = Math.max(...prices, 0);
        const binSize = Math.ceil((maxPrice / 10) / 50) * 50 || 50;
        const bins = {};
        for (let i = 0; i < Math.ceil(maxPrice / binSize) + 1; i++) {
            const binStart = i * binSize;
            bins[`${binStart}-${binStart + binSize}`] = 0;
        }
        prices.forEach(price => {
            const binIndex = Math.floor(price / binSize);
            const binStart = binIndex * binSize;
            const binLabel = `${binStart}-${binStart + binSize}`;
            if (bins[binLabel] !== undefined) bins[binLabel]++;
        });
        priceChart.data = {
            labels: Object.keys(bins),
            datasets: [{ label: 'Product Count', data: Object.values(bins), backgroundColor: '#818CF8' }]
        };
        priceChart.update();
    }
    
    function addEventListeners() {
        const handleFilterChange = () => {
            clearTimeout(filterTimeout);
            filterTimeout = setTimeout(fetchAndRenderProducts, 300); // Debounce
        };

        elements.searchInput.addEventListener('input', () => {
            state.filters.search = elements.searchInput.value.toLowerCase();
            handleFilterChange();
        });

        elements.categoryFilters.addEventListener('change', (e) => {
            if (e.target.type === 'checkbox') {
                const category = e.target.dataset.category;
                e.target.checked ? state.filters.categories.add(category) : state.filters.categories.delete(category);
                handleFilterChange();
            }
        });
        
        const priceHandler = () => {
            state.filters.priceMin = parseFloat(elements.priceMinInput.value) || null;
            state.filters.priceMax = parseFloat(elements.priceMaxInput.value) || null;
            handleFilterChange();
        };
        elements.priceMinInput.addEventListener('change', priceHandler);
        elements.priceMaxInput.addEventListener('change', priceHandler);
        
        elements.sortBySelect.addEventListener('change', () => {
            state.sortBy = elements.sortBySelect.value;
            sortProducts();
            renderProductGrid(state.products);
        });
    }
    
    function renderProductGrid(products) {
        elements.count.textContent = products.length;
        const showNoResults = products.length === 0;
        elements.grid.classList.toggle('hidden', showNoResults);
        elements.noResults.classList.toggle('hidden', !showNoResults);

        elements.grid.innerHTML = products.map(p => `
            <div class="bg-gray-800/50 border border-gray-700/50 rounded-xl overflow-hidden flex flex-col transition-transform transform hover:-translate-y-1">
                <a href="${p.product_link}" target="_blank" rel="noopener noreferrer">
                    <img src="${p.image_url}" alt="${p.name}" class="w-full h-56 object-cover bg-gray-700">
                </a>
                <div class="p-4 flex flex-col flex-grow">
                    <h3 class="font-semibold text-gray-200 mb-2 flex-grow min-h-[40px] text-sm">${p.name}</h3>
                    <div class="flex items-center justify-between mb-3">
                        <p class="text-xl font-bold text-white">${p.price ? p.price.toFixed(2) : 'N/A'} <span class="text-sm font-normal text-gray-400">${p.currency}</span></p>
                        ${p.discount_percentage ? `<span class="bg-red-500/20 text-red-400 text-xs font-bold px-2 py-1 rounded">${p.discount_percentage}% OFF</span>` : ''}
                    </div>
                    <div class="flex flex-wrap gap-2 text-xs mt-auto pt-2 border-t border-gray-700/50">
                        ${p.category ? `<span class="bg-indigo-500/20 text-indigo-300 px-2 py-1 rounded-full">${p.category}</span>` : ''}
                        ${p.colour ? `<span class="bg-blue-500/20 text-blue-300 px-2 py-1 rounded-full">${p.colour}</span>` : ''}
                        ${p.material ? `<span class="bg-green-500/20 text-green-300 px-2 py-1 rounded-full">${p.material}</span>` : ''}
                    </div>
                </div>
            </div>
        `).join('');
    }

    init();
});
```

---
**File: `data/products.json`**
```json
[
  {
    "product_id": "N40893810A",
    "name": "Classic Crew Neck T-Shirt",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 79.00,
    "currency": "AED",
    "discount_percentage": 20,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/428338/pexels-photo-428338.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Clothing",
    "colour": "White",
    "material": "Cotton",
    "scraped_at": "2025-06-22T22:10:00Z"
  },
  {
    "product_id": "Z1E2844DB0",
    "name": "June Embroidered Loose Fit Shirt",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 189.00,
    "currency": "AED",
    "discount_percentage": null,
    "delivery_type": "Market",
    "image_url": "https://images.pexels.com/photos/769749/pexels-photo-769749.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Clothing",
    "colour": "Blue",
    "material": "Linen",
    "scraped_at": "2025-06-22T22:15:00Z"
  },
  {
    "product_id": "N53346754A",
    "name": "Air Zoom Pegasus Running Shoes",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 450.00,
    "currency": "AED",
    "discount_percentage": 15,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/1032110/pexels-photo-1032110.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Shoes",
    "colour": "Black",
    "material": "Mesh",
    "scraped_at": "2025-06-22T21:05:00Z"
  },
  {
    "product_id": "N12345678B",
    "name": "Leather Crossbody Bag",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 320.50,
    "currency": "AED",
    "discount_percentage": null,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/1152077/pexels-photo-1152077.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Bags",
    "colour": "Brown",
    "material": "Leather",
    "scraped_at": "2025-06-21T18:30:00Z"
  },
  {
    "product_id": "N98765432C",
    "name": "Wireless Noise Cancelling Headphones",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 899.00,
    "currency": "AED",
    "discount_percentage": 10,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/3945683/pexels-photo-3945683.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Electronics",
    "colour": "Silver",
    "material": "Plastic",
    "scraped_at": "2025-06-22T20:00:00Z"
  },
  {
    "product_id": "N24681357D",
    "name": "Slim Fit Denim Jeans",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 250.00,
    "currency": "AED",
    "discount_percentage": 30,
    "delivery_type": "Market",
    "image_url": "https://images.pexels.com/photos/52573/jeans-pants-blue-shop-52573.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Clothing",
    "colour": "Blue",
    "material": "Denim",
    "scraped_at": "2025-06-22T19:45:00Z"
  },
  {
    "product_id": "N13579246E",
    "name": "Classic Leather Loafers",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 375.00,
    "currency": "AED",
    "discount_percentage": null,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/267320/pexels-photo-267320.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Shoes",
    "colour": "Brown",
    "material": "Leather",
    "scraped_at": "2025-06-21T14:20:00Z"
  },
  {
    "product_id": "N11223344F",
    "name": "Smartwatch with Heart Rate Monitor",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 1250.00,
    "currency": "AED",
    "discount_percentage": 5,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/277406/pexels-photo-277406.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Electronics",
    "colour": "Black",
    "material": "Aluminum",
    "scraped_at": "2025-06-22T23:01:00Z"
  },
  {
    "product_id": "N55667788G",
    "name": "Canvas Travel Backpack",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 199.99,
    "currency": "AED",
    "discount_percentage": 25,
    "delivery_type": "Market",
    "image_url": "https://images.pexels.com/photos/1545998/pexels-photo-1545998.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Bags",
    "colour": "Khaki",
    "material": "Canvas",
    "scraped_at": "2025-06-20T11:00:00Z"
  },
  {
    "product_id": "N99887766H",
    "name": "Floral Print Summer Dress",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 299.00,
    "currency": "AED",
    "discount_percentage": null,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/1755428/pexels-photo-1755428.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Clothing",
    "colour": "Multicolor",
    "material": "Viscose",
    "scraped_at": "2025-06-22T15:15:00Z"
  },
  {
    "product_id": "N12121212I",
    "name": "High-Top Basketball Sneakers",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 620.00,
    "currency": "AED",
    "discount_percentage": 10,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/1661471/pexels-photo-1661471.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Shoes",
    "colour": "Red",
    "material": "Synthetic",
    "scraped_at": "2025-06-22T22:50:00Z"
  },
  {
    "product_id": "N34343434J",
    "name": "Portable Bluetooth Speaker",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 250.00,
    "currency": "AED",
    "discount_percentage": 15,
    "delivery_type": "Market",
    "image_url": "https://images.pexels.com/photos/1279929/pexels-photo-1279929.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Electronics",
    "colour": "Blue",
    "material": "Fabric",
    "scraped_at": "2025-06-22T10:30:00Z"
  },
  {
    "product_id": "N56565656K",
    "name": "Wool Blend Peacoat",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 550.00,
    "currency": "AED",
    "discount_percentage": null,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/1254502/pexels-photo-1254502.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Clothing",
    "colour": "Navy",
    "material": "Wool",
    "scraped_at": "2025-06-22T09:00:00Z"
  },
  {
    "product_id": "N78787878L",
    "name": "Minimalist Leather Tote Bag",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 450.00,
    "currency": "AED",
    "discount_percentage": 20,
    "delivery_type": "Express",
    "image_url": "https://images.pexels.com/photos/2905238/pexels-photo-2905238.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Bags",
    "colour": "Black",
    "material": "Leather",
    "scraped_at": "2025-06-21T23:55:00Z"
  },
  {
    "product_id": "N90909090M",
    "name": "Ergonomic Office Chair",
    "product_link": "https://www.noon.com/uae-en/p/",
    "price": 750.00,
    "currency": "AED",
    "discount_percentage": null,
    "delivery_type": "Market",
    "image_url": "https://images.pexels.com/photos/7210747/pexels-photo-7210747.jpeg?auto=compress&cs=tinysrgb&w=600",
    "category": "Furniture",
    "colour": "Gray",
    "material": "Mesh",
    "scraped_at": "2025-06-19T13:00:00Z"
  }
]
```