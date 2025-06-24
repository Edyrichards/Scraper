# 🚀 Beginner's Guide: Setting Up Noon Scraper & Analyzer on macOS

Welcome! This guide will walk you through setting up the Noon.com Scraper & Analyzer application on your Mac, step by step. We'll start from the very beginning, and no prior technical experience is needed. Let's get started!

### **What We'll Accomplish**

By the end of this guide, you will have:
1.  Installed all the necessary tools (Homebrew, Python, Redis).
2.  Set up the project and its dependencies in a clean, isolated environment.
3.  Successfully launched the three components of the application.

---

### **Part 1: Installing the Foundation Tools**

Before we can run the application, we need to install a few key pieces of software. We'll use a tool called **Homebrew**, which is a package manager for macOS. Think of it as an App Store for developers that makes installing software from the command line incredibly easy.

#### **Step 1.1: Install Homebrew**

First, open the **Terminal** application on your Mac. You can find it in `Applications -> Utilities`, or by searching for it with Spotlight (⌘ + Space).

Once your terminal is open, copy and paste the following command, then press **Enter**.

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
> **What does this do?** This command securely downloads and runs the official Homebrew installation script. It might ask for your Mac's password to proceed. You'll see a lot of text scroll by – this is normal! It's just Homebrew setting itself up.

#### **Step 1.2: Install Python 3**

While macOS comes with a version of Python, we need a more recent one. We'll use Homebrew to install it.

In the same terminal window, run:

```shell
brew install python
```
> **What does this do?** This tells Homebrew to download and install the latest stable version of Python 3. Python is the programming language the entire backend of our application is written in.

#### **Step 1.3: Install Redis**

Redis is a high-speed in-memory database. In our application, it acts as a "message broker" or a central hub. The web server sends scraping jobs to this hub, and our background worker picks them up from there.

Install Redis with this command:

```shell
brew install redis
```
> **What does this do?** This command instructs Homebrew to download and install the Redis server on your Mac. We'll start it up later when we're ready to run the app.

---

### **Part 2: Setting Up the Project**

Now that we have our tools, let's get the application code and prepare it to run.

#### **Step 2.1: Clone the Project Code**

We'll download the project from its repository using Git. If you don't have Git, Homebrew might have installed it for you. You can check with `git --version`.

In your terminal, run the following commands.

```shell
# Clone the project repository (replace with the actual URL)
git clone <your-repository-url>

# Navigate into the newly created project folder
cd noon-scraper-analyzer
```
> **What does this do?** The `git clone` command downloads a full copy of the project's code to your computer. The `cd` (change directory) command moves your terminal's focus into that new folder, so all subsequent commands are run from inside the project.

#### **Step 2.2: Create a Virtual Environment**

This is a very important step. We will create a *virtual environment*, which is an isolated, private workspace for this project. It ensures that the specific versions of software packages we install for this project don't interfere with any other projects on your computer.

```shell
python3 -m venv venv
```
> **What does this do?** This command uses the Python 3 we installed to create a virtual environment named `venv` inside our project directory.

#### **Step 2.3: Activate the Virtual Environment**

Now, we need to "turn on" or activate this environment.

```shell
source venv/bin/activate
```
> **What will you see?** After running this, you'll notice that your terminal prompt changes, and now has `(venv)` at the beginning. This is how you know the virtual environment is active!
>
> **(venv) your-mac-name:noon-scraper-analyzer your-username$**
>
> *Remember: You must activate the virtual environment every time you open a new terminal window to work on this project.*

#### **Step 2.4: Install All Required Packages**

The project comes with a file called `requirements.txt`. This is a shopping list of all the Python packages the application needs to function. We'll use `pip`, Python's package installer, to install them all at once.

```shell
pip install -r requirements.txt
```
> **What does this do?** `pip` reads every line in the `requirements.txt` file and installs the specified package into our active `(venv)` environment.

Next, we need to install the browser engines for Playwright, the tool our scraper uses to control a web browser.

```shell
playwright install
```
> **What does this do?** This command downloads the specific, headless browser versions (like Chrome, Firefox) that Playwright needs to do its job of scraping websites.

---

### **Part 3: Running the Application**

The application is made of three services that must run at the same time in **three separate terminal windows**.



Let's launch them one by one.

#### **Terminal Window 1: Start the Redis Server**

This is our message hub. It needs to be running first so the other services can connect to it.

1.  Open a **new** terminal window.
2.  Run the following command:

```shell
redis-server
```
> **What will you see?** Redis will start and display a large logo and some log information. **Just leave this terminal window open and running.** If you close it, the application will stop working.

#### **Terminal Window 2: Start the Celery Worker**

This is the "heavy lifter" that runs in the background to perform the scraping and analysis tasks.

1.  Open a **second new** terminal window.
2.  Navigate to your project folder: `cd path/to/noon-scraper-analyzer`
3.  **Activate the virtual environment**: `source venv/bin/activate`
4.  Now, start the Celery worker:

```shell
celery -A tasks.celery_app worker --loglevel=info
```
> **What will you see?** The Celery worker will start up and show a "Ready" message. This window is where you can monitor the progress of scraping tasks. **Leave this terminal running as well.**

#### **Terminal Window 3: Start the FastAPI Web Server**

This is the final piece! This server runs the web interface you'll interact with and the API that serves data to the dashboard.

1.  Open a **third new** terminal window.
2.  Navigate to your project folder: `cd path/to/noon-scraper-analyzer`
3.  **Activate the virtual environment**: `source venv/bin/activate`
4.  Finally, start the web server:

```shell
uvicorn main:app --reload
```
> **What will you see?** Uvicorn will start and tell you that the application is running on `http://127.0.0.1:8000`.

### ✅ **You're All Set!**

Congratulations! The entire application is now up and running on your Mac.

-   Open your web browser (like Chrome or Safari) and go to: **`http://127.0.0.1:8000`**
-   You should see the application's homepage. You can now paste a Noon.com URL and start analyzing!

To shut down the application, go to each of the three terminal windows and press **Ctrl + C**.