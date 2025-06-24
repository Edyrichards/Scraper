# **Design Document: Noon.com Scraper & Analytics Web Application**

**Version:** 1.0  
**Date:** June 22, 2025  
**Author:** Expert Analyst

---

### **1. Application Architecture**

The proposed architecture transforms the existing command-line tool into a robust, scalable, and user-friendly web application. It decouples the user-facing web interface from the resource-intensive scraping and AI classification tasks to ensure a responsive user experience.

#### **1.1. System Overview**

The system is designed with a modern microservices-oriented approach, consisting of three primary components:

1.  **Frontend (Client)**: A dynamic single-page application (SPA) built with **React**, providing an interactive dashboard for initiating scrapes and analyzing data.
2.  **Backend (API Server)**: A lightweight API server built with **FastAPI** (Python), responsible for handling user requests, managing job lifecycles, and serving processed data.
3.  **Task Queue & Workers**: A distributed task queue system using **Celery** and **Redis**, which executes the long-running scraping (Playwright) and AI classification (CLIP) jobs asynchronously in the background.

#### **1.2. Architectural Diagram**

![A diagram showing the web application architecture. A 'User Browser (React)' box connects to a 'Backend API (FastAPI)' via REST API calls. The FastAPI server interacts with a 'PostgreSQL Database' for reading/writing product data and with a 'Redis' instance which serves as both a message broker and a results backend. The Redis broker passes jobs to 'Celery Worker(s)'. The Celery Worker box contains two sub-processes: 'Scraper (Playwright)' and 'AI Classifier (CLIP)'. The Celery Worker also interacts with the PostgreSQL Database to store the results of its tasks.](https://i.imgur.com/GzB9tqf.png)

#### **1.3. Asynchronous Task Management**

To prevent blocking the API server and timing out user requests, all scraping and classification jobs are handled asynchronously.

*   **Job Initiation**: When a user submits a URL via the `POST /api/scrape` endpoint, the FastAPI server *does not* start scraping immediately. Instead, it creates a task record in the database, pushes a job message to the Redis message broker, and instantly returns a `job_id` to the user.
*   **Job Execution**: One or more Celery workers, running as separate processes, listen for jobs on the Redis queue. A worker picks up the job, executes the Playwright scraping and CLIP classification logic from the original scripts, and updates the PostgreSQL database with the results.
*   **Status Tracking**: The frontend can periodically poll the `GET /api/status/{job_id}` endpoint. The FastAPI server checks the status of the job (e.g., `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`) from the Celery results backend (Redis) or the database and reports back to the client.

This architecture ensures the web application remains responsive regardless of the number or duration of scraping tasks being executed. It is also horizontally scalable by simply adding more Celery worker instances.

---

### **2. Database Schema**

A relational database (e.g., PostgreSQL) is recommended for storing the structured product data. The schema is designed to be normalized and efficient for querying and filtering.

#### **2.1. Table: `products`**

This table will store all data points for each unique product scraped from Noon.com. The `product_id` extracted from the URL serves as the natural primary key.

| Column Name | Data Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `product_id` | `VARCHAR(30)` | `PRIMARY KEY` | The unique 20-25 character identifier from Noon.com. |
| `name` | `TEXT` | `NOT NULL` | The full name of the product. |
| `product_url` | `TEXT` | `NOT NULL`, `UNIQUE` | The full, direct URL to the product page. |
| `price` | `DECIMAL(10, 2)` | `NOT NULL` | The current selling price. |
| `currency` | `VARCHAR(10)` | | The currency of the price (e.g., "AED"). |
| `discount_percentage` | `INTEGER` | | The discount percentage, if available. |
| `delivery_type` | `VARCHAR(255)` | | The type of delivery offered (e.g., "Express"). |
| `image_url` | `TEXT` | | The direct URL to the main product image. |
| `category` | `VARCHAR(100)` | | AI-classified main category (e.g., Clothing, Shoes). |
| `colour` | `VARCHAR(100)` | | AI-classified primary color. |
| `material` | `VARCHAR(100)` | | AI-classified primary material. |
| `pattern` | `VARCHAR(100)` | | AI-classified visual pattern. |
| `occasion` | `VARCHAR(100)` | | AI-classified suitable occasion. |
| `garment_type` | `VARCHAR(100)` | | AI-classified clothing-specific type. |
| `shoe_type` | `VARCHAR(100)` | | AI-classified shoe-specific type. |
| `bag_type` | `VARCHAR(100)` | | AI-classified bag-specific type. |
| `scraped_at` | `TIMESTAMPTZ` | `NOT NULL` | The timestamp (with time zone) when the data was scraped. |
| `last_updated_at` | `TIMESTAMPTZ` | `NOT NULL` | The timestamp of the last update to this record. |

#### **2.2. SQL `CREATE TABLE` Statement**

```sql
CREATE TABLE products (
    product_id VARCHAR(30) PRIMARY KEY,
    name TEXT NOT NULL,
    product_url TEXT NOT NULL UNIQUE,
    price DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(10),
    discount_percentage INTEGER,
    delivery_type VARCHAR(255),
    image_url TEXT,
    category VARCHAR(100),
    colour VARCHAR(100),
    material VARCHAR(100),
    pattern VARCHAR(100),
    occasion VARCHAR(100),
    garment_type VARCHAR(100),
    shoe_type VARCHAR(100),
    bag_type VARCHAR(100),
    scraped_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Optional: Create an index for faster filtering by category and price
CREATE INDEX idx_products_category_price ON products (category, price);
```

---

### **3. API Endpoints**

The RESTful API provides the interface between the frontend and the backend services.

#### **`POST /api/scrape`**
*   **Description**: Initiates a new asynchronous scraping and classification job.
*   **Request Body**:
    ```json
    {
      "url": "https://www.noon.com/uae-en/fashion/women"
    }
    ```
*   **Success Response (202 Accepted)**:
    ```json
    {
      "job_id": "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890",
      "status": "QUEUED",
      "message": "Scraping job has been successfully queued."
    }
    ```
*   **Error Response (400 Bad Request)**: If the URL is invalid.

#### **`GET /api/status/{job_id}`**
*   **Description**: Checks the status of a specific scraping job.
*   **URL Parameter**: `job_id` (string) - The ID returned from the `/api/scrape` call.
*   **Success Response (200 OK)**:
    ```json
    {
      "job_id": "a1b2c3d4-e5f6-7890-a1b2-c3d4e5f67890",
      "status": "COMPLETED", // Can be QUEUED, RUNNING, COMPLETED, FAILED
      "progress": {
        "found": 150,
        "processed": 150
      },
      "created_at": "2025-06-22T23:30:00Z",
      "finished_at": "2025-06-22T23:35:10Z"
    }
    ```

#### **`GET /api/products`**
*   **Description**: Retrieves a paginated and filterable list of all scraped products.
*   **Query Parameters**:
    *   `limit` (int, default: 20): Number of items per page.
    *   `page` (int, default: 1): The page number to retrieve.
    *   `category` (string): Filter by a specific AI-classified category (e.g., `Clothing`).
    *   `price_min` (float): Filter for products with a price greater than or equal to this value.
    *   `price_max` (float): Filter for products with a price less than or equal to this value.
    *   `sort_by` (string, e.g., `price`, `scraped_at`): Field to sort by.
    *   `order` (string, `asc` or `desc`, default: `desc`): Sort order.
*   **Success Response (200 OK)**:
    ```json
    {
      "pagination": {
        "total_items": 1250,
        "total_pages": 63,
        "current_page": 1,
        "limit": 20
      },
      "data": [
        {
          "product_id": "N40893810A",
          "name": "Classic Crew Neck T-Shirt",
          "price": 79.00,
          "image_url": "https://cdn.noon.com/...",
          "category": "Clothing",
          "colour": "White",
          /* ... other fields ... */
        }
      ]
    }
    ```

---

### **4. User Interface (UI) and User Experience (UX) Flow**

The UI will be clean, intuitive, and focused on making the scraping process and data analysis as simple as possible.

#### **4.1. Homepage / Initiation Page**

*   **Layout**: A minimalist full-screen layout with a central focus.
*   **Components**:
    1.  **Headline**: A clear, concise headline (e.g., "Analyze Noon.com Products Instantly").
    2.  **Input Field**: A single, large text input field prominently displayed, with placeholder text like `Paste a Noon.com category or search URL here`.
    3.  **Primary Button**: A large, inviting "Start Analysis" button next to or below the input field.
*   **Experience**: The design avoids clutter, guiding the user to the single most important action.

#### **4.2. Process Flow**

1.  **Submission**: User pastes a valid Noon.com URL into the input field and clicks "Start Analysis".
2.  **Feedback**:
    *   The "Start Analysis" button becomes disabled and shows a loading spinner to indicate processing.
    *   An API call is made to `POST /api/scrape`.
3.  **Redirection**: Upon receiving a successful `202 Accepted` response with a `job_id`, the application automatically redirects the user to the main Analytics Dashboard.
4.  **Notification**: A non-intrusive notification (a "toast" message) appears on the dashboard, stating:
    > "✅ Scraping job started! New products will appear on the dashboard as they are processed."
5.  **Live Updates**: The dashboard will now contain the logic to periodically check the job status and refresh the product list automatically once the job is complete, providing a seamless experience without requiring a manual page reload.

---

### **5. Analytics Dashboard Design**

The dashboard is the central hub for viewing, filtering, and understanding the scraped data.

#### **5.1. Layout**

A professional, two-column layout will be used:
*   **Left Sidebar (Filters)**: A fixed-width sidebar on the left containing all interactive filtering and sorting controls.
*   **Main Content Area (Results)**: A wider main area on the right that displays the product data through cards, tables, and charts.

#### **5.2. Components**

**1. Product Cards (Default View)**
Each product will be represented by a clean, modern card containing:
*   **Product Image**: The primary `image_url` displayed prominently at the top.
*   **Product Name**: The full product `name`.
*   **Price**: The `price` and `currency` clearly displayed.
*   **AI Attributes**: Key classified attributes like `category` and `colour` shown as small, colored tags or badges for quick visual scanning.
*   **Link**: The entire card is clickable, opening the `product_url` in a new tab.

**2. Data Table (Toggleable View)**
*   A "View as Table" toggle switch will be present in the main content header.
*   When activated, it replaces the product cards with a dense, sortable data table.
*   **Columns**: Includes `Name`, `Price`, `Category`, `Colour`, `Material`, `Scraped At`, etc.
*   **Features**: Supports column sorting and pagination controls at the bottom.

**3. Visualizations**
Two key charts will be displayed at the top of the main content area to provide high-level insights.
*   **Bar Chart: Product Count by Category**: A vertical bar chart showing the total number of products found for each AI-classified category (e.g., Clothing, Shoes, Bags). This gives an immediate overview of the product mix.
*   **Histogram: Price Distribution**: A histogram showing the frequency of products across different price buckets (e.g., 0-50 AED, 51-100 AED, 101-150 AED). This helps in understanding the pricing strategy of the scraped category.

**4. Interactive Filters (Sidebar)**
The left sidebar will contain the following controls, which will re-query the `GET /api/products` endpoint on change:
*   **Search Bar**: A text input to filter products by name.
*   **Category Filter**: A list of checkboxes for each available AI-classified category, allowing multi-selection.
*   **Price Range Slider**: A dual-handle slider to set a minimum and maximum price.
*   **Date Range Picker**: A calendar control to filter products scraped within a specific time frame.
*   **Sort Options**: A dropdown menu to sort by `Price (High to Low)`, `Price (Low to High)`, or `Date Scraped (Newest First)`.