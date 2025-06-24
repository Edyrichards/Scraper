# **Project Delivery Document: Noon.com Scraper & Analyzer**

**Version:** 2.0 (Final)
**Date:** 23/06/2025
**Status:** Completed & Delivered

---

### **1. Project Overview**

This document provides the definitive guide and formal handover for the **Noon.com Scraper & Analyzer** web application. The application is a sophisticated tool designed to automate the collection, enrichment, and analysis of product data from Noon.com.

The core workflow allows a user to submit a Noon.com category or search URL through a minimalist web interface. This action triggers a robust, asynchronous backend process that scrapes product listings, uses an AI model to classify product images, and persists the structured data into a database. The results are then presented on a dynamic, interactive analytics dashboard where the user can filter, sort, and visualize the data to derive market insights.



---

### **2. Technology Stack**

The application is built on a modern technology stack, chosen for performance, scalability, and maintainability.

| Component | Technology / Library | Role & Justification |
| :--- | :--- | :--- |
| **Frontend** | React (or Vanilla JS) | Provides the user interface, including the URL submission page and the interactive analytics dashboard. |
| | TailwindCSS | A utility-first CSS framework used for rapid and responsive UI design. |
| | Chart.js | Renders responsive, interactive charts for data visualization (e.g., category distribution, price histograms). |
| | Lucide Icons | Provides a clean and consistent set of icons for the UI. |
| **Backend API**| Python 3.9+ | Core programming language for the backend. |
| | FastAPI | A high-performance web framework for building the RESTful API endpoints that serve data to the frontend. |
| **Asynchronous Tasks**| Celery | A distributed task queue system that manages long-running jobs (scraping, AI classification) in the background. |
| | Redis | Serves as the high-speed message broker for Celery, facilitating communication between the API server and the workers. |
| **Data Storage**| PostgreSQL (or SQLite) | The relational database for persisting all scraped and classified product data. SQLAlchemy is used as the ORM. |
| **Scraping Engine**| Playwright | A modern web automation library used for reliably scraping dynamic, JavaScript-heavy websites like Noon.com. |
| | BeautifulSoup | A library for efficiently parsing the HTML content retrieved by Playwright. |
| **AI Classification**| PyTorch | The underlying deep learning framework for the AI model. |
| | OpenAI CLIP (`ViT-B/32`) | A state-of-the-art vision-language model used for zero-shot classification of product images into various categories and attributes. |

---

### **3. Features in Detail**

#### **3.1. URL Submission & Asynchronous Scraping**

The user journey begins on a simple landing page where they can initiate the entire process.

1.  **Submission**: The user pastes a Noon.com category or search results URL into the input field and clicks "Start Analysis".
2.  **Job Initiation**: The frontend sends a `POST` request to the `/api/scrape` endpoint on the FastAPI server.
3.  **Task Queuing**: The FastAPI server *does not* perform the scraping directly. Instead, it creates a new job definition and pushes it onto the Redis message queue. It then immediately returns a `202 Accepted` response to the user, confirming the task has been queued. This ensures the UI remains fast and responsive.
4.  **Background Execution**: A separate **Celery Worker** process, constantly listening to the Redis queue, picks up the job. The worker executes the scraping logic using Playwright to navigate the URL, scroll to load all products, and extract the raw data.
5.  **Dashboard Redirection**: Upon successful job submission, the user is automatically redirected to the analytics dashboard to view the results as they become available.

#### **3.2. AI-Powered Product Classification**

After scraping, the raw product data is enriched using OpenAI's CLIP model. This process occurs within the same background Celery task.

1.  **Image Analysis**: For each scraped product, the system uses its `image_url` to download the product image.
2.  **Zero-Shot Classification**: The CLIP model compares the image against a predefined set of text-based labels (e.g., "a photo of clothing," "a photo of shoes"). This is done without any model retraining.
3.  **Hierarchical Tagging**: The classification is performed hierarchically:
    *   First, the main **category** (e.g., *Clothing*, *Shoes*, *Bags*) is determined.
    *   Based on the main category, more specific attributes like **garment type** (*Shirt*, *Dress*) or **shoe type** (*Sneakers*, *Boots*) are classified.
    *   General attributes like **color**, **material**, and **pattern** are also classified.
4.  **Data Enrichment**: The resulting AI-generated tags are added to the product's data record before it is saved to the database.

#### **3.3. Interactive Analytics Dashboard**

The dashboard is the central hub for data analysis, featuring a sleek, responsive design.

![A screenshot of the final analytics dashboard, showing a dark theme. On the left is a sidebar with filters for search, category, and price. The main content area shows two charts at the top (Products by Category, Price Distribution) and a grid of product cards below.](https://i.imgur.com/GzB9tqf.png)

**Key Components:**
*   **Visualizations**: At the top of the dashboard, two `Chart.js` charts provide an instant overview:
    *   **Products by Category**: A bar chart showing the distribution of products across different AI-classified categories.
    *   **Price Distribution**: A histogram illustrating how many products fall into different price buckets.
*   **Product Grid**: The main area displays all collected products as individual cards, each showing the product image, name, price, and key AI-classified tags.
*   **Dynamic Counts**: The dashboard header dynamically updates to show the total number of products currently being displayed based on the active filters.

#### **3.4. Server-Side Filtering**

To ensure performance with large datasets, all filtering is handled on the server. The dashboard does *not* load all products at once.

*   **How it Works**: When a user changes a filter (e.g., checks a category box, enters a price range), the frontend JavaScript constructs a new API request to the `GET /api/products` endpoint with the filter criteria as query parameters.
    *   *Example Request*: `GET /api/products?category=Clothing&min_price=100&max_price=500`
*   **Backend Logic**: The FastAPI backend receives this request, and the SQLAlchemy query builder dynamically adds `WHERE` clauses to the database query based on the provided parameters. This ensures the database does the heavy lifting.
*   **Efficient Rendering**: The server returns only the products that match the filters, which the frontend then renders. This approach is highly efficient and scales well, as the browser only ever has to handle a manageable subset of the total data.

**Available Filters:**
*   **Search**: Full-text search on product names.
*   **Category**: Multi-select checkboxes for all available AI-classified categories.
*   **Brand**: A text input to filter by brand name (searches within the product title).
*   **Price Range**: Min/Max number inputs to define a price window.

---

### **4. Updated README.md (Setup & Deployment Guide)**

This section provides a complete, standalone guide for setting up and running the application in a development environment.

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