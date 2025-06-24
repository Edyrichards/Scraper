### **Technical Specification: Noon.com Product Scraper & AI Classifier**

This document outlines the technical architecture, data flow, and core components of the Noon.com product scraping and classification pipeline. It serves as a foundational blueprint for development and maintenance.

---

### **1. Overall Architecture**

The system is designed as a modular command-line application that orchestrates a multi-stage process: web scraping, data processing, optional AI-powered classification, and data exporting.

**High-Level Workflow:**

The process is initiated via a Command-Line Interface (CLI), which acts as the main controller. The workflow proceeds as follows:

1.  **Initiation**: The user executes a command via `main.py`, specifying the target Noon.com URL and desired operations (e.g., scrape, classify, filter).
2.  **Scraping**: The Scraper module, utilizing Playwright, launches a headless browser to navigate to the URL, handle dynamic content (like infinite scroll), and extract the raw HTML of product listings.
3.  **Parsing & Structuring**: Raw product data (name, price, image URL, etc.) is parsed from the HTML and structured, typically into a list of dictionaries or a pandas DataFrame.
4.  **AI Classification (Optional)**: If enabled, each product's image URL is passed to the AI Classifier module. The CLIP model analyzes the image against a predefined taxonomy of attributes (e.g., color, material, pattern) to enrich the product data.
5.  **Processing & Export**: The final, structured data (either raw or AI-enriched) is passed to the Data Processor, which formats it and exports it into user-specified file formats (CSV, Excel, JSON).

**Architectural Diagram:**

![A flowchart showing the system architecture. It starts with a 'CLI (main.py)' box. An arrow points to 'Scraper (Playwright & BeautifulSoup)' which extracts data. From the scraper, an arrow points to a 'Data Structuring (Pandas DataFrame)' step. From there, two paths diverge. One path goes to an optional 'AI Classifier (CLIP Model)' which enriches the data with attributes. Both the optional path and the direct path from data structuring lead to the 'Data Processor & Exporter'. Finally, arrows point from the exporter to three output formats: CSV, Excel, and JSON.](https://i.imgur.com/vHqQ9jK.png)

---

### **2. Core Scraping Logic**

The scraping mechanism is designed to handle the dynamic, JavaScript-heavy nature of modern e-commerce websites like Noon.com.

**Primary Technologies:**
*   **Browser Automation**: `Playwright` is used to control a real browser engine (Chromium). This is essential for correctly rendering pages that rely on JavaScript for content loading, including product listings populated via API calls or "infinite scroll" user interactions.
*   **HTML Parsing**: `BeautifulSoup` is used to parse the static HTML content retrieved by Playwright, enabling robust extraction of data using CSS selectors.

**Data Extraction Process:**

1.  **URL Input**: The scraper accepts any Noon.com UAE URL, including category pages, search results, or specific brand pages.
2.  **Page Navigation**: Playwright navigates to the URL and systematically scrolls down the page to trigger the loading of all products. A configurable pause is used to allow content to render.
3.  **Product Link Identification**: The core of the extraction logic is identifying valid product links. The system uses a specific regular expression to ensure only correct product page URLs are captured.
    *   **URL Pattern:** `re.compile(r'/uae-en/[^/]+/([A-Z0-9]{20,25})/p/')`
    *   This pattern correctly identifies the unique, uppercase alphanumeric **Product ID** (e.g., `Z1E2844DB05FD558D1DDBZ`), which is crucial for forming a direct, functional link to the product page.
4.  **Data Point Extraction**: For each product identified on the listing page, the following key data points are extracted from the HTML structure:

| Data Point | Example | Extraction Logic |
| :--- | :--- | :--- |
| `name` | "June Embroidered Loose Fit Shirt" | Extracted from the primary product title element. |
| `product_link` | "https://www.noon.com/.../Z1E2844DB.../p/" | Constructed using the base URL and the matched `href`. |
| `product_id` | `Z1E2844DB05FD558D1DDBZ` | Captured from the URL using the regex group. |
| `price` | "189.00 AED" | Scraped from the price display element. |
| `discount` | "25% OFF" | Scraped from the discount badge, if present. |
| `delivery_type` | "Express Delivery" | Extracted from shipping information tags. |
| `image_url` | "https://cdn.noon.com/images/..." | Extracted from the `src` attribute of the product's `<img>` tag. |

---

### **3. Data Processing & Storage**

Once raw data is scraped, it is processed into a clean, structured format suitable for analysis and export.

**Data Structuring:**
*   The primary tool for in-memory data management is the **pandas DataFrame**. This structure provides a powerful and efficient way to handle tabular data, perform cleaning operations, and facilitate easy export to multiple formats.

**Proposed Data Schema:**

The following schema defines the comprehensive data structure for each product, incorporating both scraped and AI-classified attributes. This schema can be directly mapped to a JSON object, database table, or spreadsheet.

| Field Name | Data Type | Description | Source |
| :--- | :--- | :--- | :--- |
| **`product_id`** | `String` | **Primary Key.** The unique 20-25 character identifier for the product. | Scraper |
| `name` | `String` | The full name of the product. | Scraper |
| `product_link` | `String` | The full, direct URL to the product page on Noon.com. | Scraper |
| `price` | `Float` | The current selling price of the product. | Scraper |
| `currency` | `String` | The currency of the price (e.g., "AED"). | Scraper/Processor |
| `discount_percentage` | `Integer` | The discount percentage, if available. | Scraper/Processor |
| `delivery_type` | `String` | The type of delivery offered (e.g., "Express"). | Scraper |
| `image_url` | `String` | The direct URL to the main product image. | Scraper |
| `category` | `String` | The main product category (e.g., Clothing, Shoes, Bags). | AI Classifier |
| `colour` | `String` | The primary color of the product. | AI Classifier |
| `material` | `String` | The primary material of the product (e.g., Cotton, Leather). | AI Classifier |
| `pattern` | `String` | The visual pattern of the product (e.g., Solid, Floral, Striped). | AI Classifier |
| `occasion` | `String` | The suitable occasion for the product (e.g., Casual, Formal). | AI Classifier |
| `garment_type` | `String` | *Clothing-specific.* Type of garment (e.g., Dress, Top, Pants). | AI Classifier |
| `shoe_type` | `String` | *Shoes-specific.* Type of shoe (e.g., Sneakers, Heels, Boots). | AI Classifier |
| `bag_type` | `String` | *Bags-specific.* Type of bag (e.g., Handbag, Backpack). | AI Classifier |
| `scraped_at` | `Timestamp` | The timestamp when the data was scraped. | Processor |

**Export Formats:**
*   **CSV**: A flattened, comma-separated file ideal for data analysis tools.
*   **Excel**: A multi-sheet `.xlsx` workbook containing the main product data on one sheet, a high-level summary on another, and attribute-specific pivot tables or breakdowns on subsequent sheets.
*   **JSON**: A structured file containing a `metadata` block (e.g., export time, total products) and a `products` array, where each element is a JSON object representing one product.

---

### **4. AI/ML Classification**

The system leverages a state-of-the-art vision-language model to automate the tedious task of product categorization and attribute tagging.

**Core Model:**
*   **OpenAI CLIP (Contrastive Language-Image Pre-Training)**: The `ViT-B/32` variant is used by default. CLIP's strength is its "zero-shot" classification capability. It can classify images based on natural language descriptions without having been explicitly trained on the specific categories beforehand.

**Classification Workflow:**

1.  **Input**: The classifier receives the `image_url` for a product.
2.  **Image Preprocessing**: The image is downloaded from the URL and preprocessed to fit the input requirements of the CLIP model.
3.  **Text Prompts**: For each attribute to be classified (e.g., "Category"), a list of possible text labels is used as prompts. For example, for "Category", the prompts would be `["a photo of clothing", "a photo of shoes", "a photo of a bag"]`.
4.  **Similarity Scoring**: CLIP computes the cosine similarity between the image embedding and the text embedding of each prompt.
5.  **Prediction**: The label corresponding to the highest similarity score is selected as the predicted attribute value. A confidence threshold (configurable in `src/config.py`) is used to filter out low-confidence predictions.
6.  **Hierarchical Classification**: The classification is done hierarchically. First, the main `category` is determined. Based on the result, a second round of classification is performed for category-specific attributes (e.g., if `category` is "Clothing", it then classifies for `garment_type`, `sleeve_type`, etc.).
7.  **Output**: The output is a dictionary of classified attributes (e.g., `{'category': 'Clothing', 'colour': 'White', 'material': 'Cotton'}`), which is then merged with the scraped data.

---

### **5. Dependencies & Environment**

To ensure the pipeline runs correctly, the following environment and dependencies are required.

**System Requirements:**
*   **Python**: Version 3.8 or newer.
*   **RAM**: A minimum of 4GB is recommended to load and run the CLIP model.
*   **Storage**: SSD storage is recommended for faster model loading from the cache.

**Key Python Libraries (`requirements.txt`):**

| Library | Purpose |
| :--- | :--- |
| `playwright` | Core library for browser automation and web scraping. |
| `beautifulsoup4` | For parsing HTML and XML documents. |
| `pandas` | For data manipulation, structuring (DataFrames), and analysis. |
| `openai-clip` | The official package for the CLIP model. |
| `torch` | The deep learning framework on which CLIP is built. |
| `torchvision` | Provides image transformations for model input. |
| `Pillow` | Python Imaging Library for opening, manipulating, and saving images. |
| `openpyxl` | Required by pandas for reading and writing Excel `.xlsx` files. |
| `pytest` | For running the automated test suite. |

**External Dependencies:**
*   **Playwright Browsers**: A browser instance must be installed via the Playwright CLI.
    ```bash
    playwright install chromium
    ```
*   **CLIP Model Files**: The first time the classifier is run, it will automatically download the CLIP model weights and cache them locally (typically in `~/.cache/clip`).

---

### **6. Key Functions and Modules**

The project is organized into a modular structure to separate concerns and improve maintainability. The core logic resides within the `src/` directory.

| Module / File | Purpose & Key Responsibilities |
| :--- | :--- |
| `main.py` / `cli.py` | **Application Entry Point & Controller.** Parses command-line arguments, orchestrates the workflow by calling the appropriate modules (scraper, classifier, processor), and handles top-level error reporting. |
| `src/scraper.py` (or `corrected_nooncom_product_scraper.py`) | **Web Scraping Logic.** Contains functions to initialize Playwright, navigate to URLs, handle dynamic page loading (scrolling), and extract raw product information using BeautifulSoup selectors. A key function would be `scrape_product_data(url: str, max_products: int)`. |
| `src/classifier.py` (or `product_attribute_classification_using_clip_vision_language_model.py`) | **AI Classification Engine.** Responsible for loading the CLIP model, preprocessing images, defining the attribute taxonomies (or loading them from config), and running the similarity comparison to classify product attributes. Key function: `classify_product_attributes(image_url: str)`. |
| `src/data_processor.py` (or `dataprocessorproductdataprocessingandexport.py`) | **Data Transformation and Export.** Takes the list of scraped/classified product dictionaries, converts it into a pandas DataFrame, performs any final cleaning or formatting, and handles the export to CSV, Excel, and JSON formats. Key functions: `export_data(products: list, formats: list)`. |
| `src/config.py` | **Central Configuration.** A non-executable file containing all key configuration variables, such as CSS selectors, model names, confidence thresholds, scraping timeouts, and file paths. This allows for easy adjustments without altering core logic. |
| `tests/` | **Test Suite.** Contains unit and integration tests (using `pytest`) for the scraper, classifier, and data processor to ensure reliability and prevent regressions. |