# **Noon.com Scraper & Analyzer**

This web application scrapes, classifies, and analyzes product data from Noon.com. It features an asynchronous task-based architecture with a FastAPI backend and a dynamic frontend dashboard.

### **Prerequisites**

Before you begin, ensure you have the following installed on your system:
*   **Python** (version 3.9 or newer)
*   **Git**
*   **Redis**: [Installation Guide](https://redis.io/docs/getting-started/installation/)

### **Step 1: Clone the Repository**

First, clone the project repository to your local machine.

```sh
git clone <your-repository-url>
cd noon-scraper-analyzer
```

### **Step 2: Set Up Python Environment**

It is highly recommended to use a virtual environment to manage project dependencies.

```sh
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

```sh
pip install -r requirements.txt
```

Next, install the necessary browser binaries for Playwright.

```sh
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

```sh
python database.py
```

### **Step 6: Running the Application**

The application consists of three main components that need to be run in separate terminal windows: **Redis**, the **Celery Worker**, and the **FastAPI Server**.

**Terminal 1: Start Redis**
If Redis is not already running as a service, start it manually:
```sh
redis-server
```

**Terminal 2: Start the Celery Worker**
With your virtual environment activated, start the Celery worker. This process will wait for and execute background jobs.
```sh
celery -A tasks.celery_app worker --loglevel=info --pool=solo
```
*Note: The `--pool=solo` flag is recommended for Windows compatibility.*

**Terminal 3: Start the FastAPI Server**
Finally, start the main web server using Uvicorn.
```sh
uvicorn main:app --reload
```

### **Step 7: Using the Application**

1.  **Open the App**: Open your web browser and navigate to `http://127.0.0.1:8000`. You will see the URL submission page.
2.  **Start Scraping**: Paste a valid Noon.com category or search URL (e.g., `https://www.noon.com/uae-en/fashion/women`) and click **"Start Analysis"**.
3.  **Monitor Progress**: You will be redirected to the dashboard. The scraping and classification will happen in the background. You can monitor the progress in the terminal where the Celery worker is running.
4.  **Analyze Data**: As products are processed and saved, they will appear on the dashboard. Use the filters on the left sidebar to search, filter by category/price, and sort the results. The charts and product grid will update in real-time based on your selections. 