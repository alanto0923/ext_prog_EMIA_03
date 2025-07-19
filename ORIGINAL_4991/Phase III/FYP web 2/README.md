# HSI DNN Alpha Yield Strategy Application

This application allows users to configure, run, and analyze the results of a Deep Neural Network (DNN) based quantitative trading strategy focused on predicting alpha yield for Hang Seng Index (HSI) constituents.

The system consists of:

1.  **Backend:** A Python application using FastAPI for the API and Celery for background task processing (running the strategy workflow). It handles data fetching, alpha calculation, model training, backtesting, and result persistence.
2.  **Frontend:** A Next.js application providing a web interface to configure runs, monitor progress, and view results (metrics, charts, logs, snapshots).

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python:** Version 3.9 or later. ([Download Python](https://www.python.org/downloads/))
*   **Node.js:** Version 18.x or later. ([Download Node.js](https://nodejs.org/))
*   **npm** or **yarn:** Package manager for Node.js (usually comes with Node.js).
*   **Redis:** A running Redis server instance. This is used as the message broker and result backend for Celery. ([Installation Guide](https://redis.io/docs/getting-started/installation/))
    *   **Easy Option (Docker):** If you have Docker installed, run:
        ```bash
        docker run -d -p 6379:6379 --name hsi-dnn-redis redis:latest
        ```
*   **(Optional) Git:** For cloning the repository if you haven't already.

## Setup Instructions

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd hsi-dnn-strategy-app
    ```

2.  **Backend Setup:**
    *   Navigate to the `backend` directory:
        ```bash
        cd backend
        ```
    *   **(Recommended)** Create and activate a Python virtual environment:
        ```bash
        # Windows
        python -m venv venv
        .\venv\Scripts\activate

        # macOS/Linux
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        ```
    *   Create a `.env` file in the `backend` directory by copying the example or creating a new one:
        ```ini
        # backend/.env

        # Celery Configuration (adjust if Redis is not on localhost)
        CELERY_BROKER_URL=redis://localhost:6379/0
        CELERY_RESULT_BACKEND=redis://localhost:6379/0

        # Base URL for accessing output files from the frontend
        # During local development (frontend on port 3000, backend on 8000):
        OUTPUT_FILE_BASE_URL=http://localhost:8000/static/output
        # Adjust if deployed or using different ports
        ```
    *   Navigate back to the project root:
        ```bash
        cd ..
        ```

3.  **Frontend Setup:**
    *   Navigate to the `frontend` directory:
        ```bash
        cd frontend
        ```
    *   Install Node.js dependencies:
        ```bash
        npm install
        # or: yarn install
        ```
    *   **(Optional but recommended)** If you intend to use `shadcn/ui` components further, initialize it (if not done during `create-next-app`):
        ```bash
        npx shadcn-ui@latest init
        # Follow the prompts
        ```
    *   Create a `.env.local` file in the `frontend` directory to tell the frontend where the backend API is running:
        ```ini
        # frontend/.env.local
        NEXT_PUBLIC_API_URL=http://localhost:8000
        ```
        *(Adjust the URL if your backend runs on a different host or port)*.
    *   Navigate back to the project root:
        ```bash
        cd ..
        ```

## Running the Application

You need to run **three** separate processes concurrently: the Redis server (if not already running as a service), the Backend Celery Worker, and the Backend FastAPI Server. Then, start the Frontend Development Server.

**1. Start Redis (if not already running):**
   * If using the Docker command from Prerequisites, it should already be running.
   * If installed manually, ensure the Redis service is started according to your OS instructions.

**2. Start the Backend Celery Worker:**
   * Open a **new terminal**.
   * Navigate to the `backend` directory: `cd backend`
   * Activate your virtual environment (if you created one):
     * Windows: `.\venv\Scripts\activate`
     * macOS/Linux: `source venv/bin/activate`
   * Run the Celery worker:
     ```bash
     * set PYTHONIOENCODING=utf-8
     * set TF_CPP_MIN_LOG_LEVEL=1
     * celery -A app.celery_app:celery_app worker --loglevel=info -P solo
     ```
     *(Keep this terminal open)*

**3. Start the Backend FastAPI Server:**
   * Open **another new terminal**.
   * Navigate to the `backend` directory: `cd backend`
   * Activate your virtual environment (if applicable).
   * Run the Uvicorn server:
     ```bash
     uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
     ```
     *(Keep this terminal open)*

**4. Start the Frontend Development Server:**
   * Open **a third new terminal**.
   * Navigate to the `frontend` directory: `cd frontend`
   * Run the Next.js development server:
     ```bash
     npm run dev
     # or: yarn dev
     ```

**5. Access the Application:**
   * Open your web browser and navigate to `http://localhost:3000`.

You should now see the frontend application running and be able to interact with it.

## How to Use

1.  **Navigate:** Open `http://localhost:3000` in your browser.
2.  **Configure:** Go to the "Configure & Run" page. Adjust the default parameters as needed (dates, tickers, K/N values, etc.).
3.  **Run:** Click the "Run Workflow" button. You'll see a notification that the run has started, and the status display will appear, polling for updates.
4.  **Monitor:** The status display will show the current step and progress percentage reported by the backend Celery task.
5.  **View Results:**
    *   Once a run successfully completes, you will be automatically redirected to the results page (`/results/<run_id>`).
    *   If you navigate away, you can find past runs on the "History" page (once implemented) or by manually going to `/results/<run_id>` if you know the ID.
    *   Use the tabs on the results page (Summary, Charts, Snapshot, Logs) to explore the output of the run.
6.  **History:** The "History" page (`/history`) lists past runs, their status, completion time, and provides links to their results pages.

## Project Structure


hsi-dnn-strategy-app/
├── frontend/ # Next.js Frontend
│ ├── app/
│ ├── components/
│ ├── lib/
│ ├── models/ # TypeScript interfaces
│ ├── public/
│ ├── .env.local # Frontend Environment Variables (API URL)
│ └── ... # Other Next.js config files
│
└── backend/ # Python Backend (FastAPI + Celery)
├── app/ # Main application code
│ ├── core/ # Core logic (strategy, config)
│ ├── models/ # Pydantic models
│ ├── routers/ # API endpoints
│ ├── services/ # Business logic layer
│ ├── utils/ # Utility functions
│ ├── celery_app.py # Celery app instance
│ ├── main.py # FastAPI app instance
│ └── tasks.py # Celery task definitions
├── output/ # Default location for run outputs & history
│ ├── <run_id_1>/ # Output files for run 1
│ ├── <run_id_2>/ # Output files for run 2
│ └── run_history.json # Persistent list of run metadata
├── venv/ # Python virtual environment (if used)
├── .env # Backend Environment Variables (Celery, Output URL)
├── requirements.txt # Python dependencies
├── Dockerfile # Optional: Dockerfile for FastAPI app
└── Dockerfile.worker # Optional: Dockerfile for Celery worker

## Stopping the Application

To stop the application, press `Ctrl+C` in each of the three terminals running the Celery worker, the Uvicorn server, and the Next.js development server. If you started Redis via Docker, you can stop and remove the container using:

```bash
docker stop hsi-dnn-redis
docker rm hsi-dnn-redis
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
Troubleshooting

CORS Errors: Ensure the origins list in backend/app/main.py includes your frontend URL (http://localhost:3000 for development) and that NEXT_PUBLIC_API_URL in frontend/.env.local points correctly to the backend.

Celery Worker Not Starting: Check that Redis is running and accessible. Verify the CELERY_BROKER_URL and CELERY_RESULT_BACKEND in backend/.env.

Task Not Running: Ensure both the Celery worker and Uvicorn server are running. Check the logs in both terminals for errors.

History Not Updating: Verify run_history.json is being created/updated in backend/output/. Check Celery worker and specific run logs for errors related to history updates or file permissions. Ensure portalocker is installed (pip install portalocker).

Type Errors: Pay close attention to tracebacks. Ensure data passed between frontend, backend API, Celery task, and strategy logic maintains the expected types (e.g., lists vs. strings, numbers vs. strings).

"Not Defined" Errors: Usually indicate missing imports in the specific Python file mentioned in the error message.

**Key points added/emphasized:**

*   Clear separation of Backend and Frontend setup.
*   Instructions for creating/activating virtual environments.
*   Explicit creation of `.env` (backend) and `.env.local` (frontend) files with example content.
*   Clear steps for running the *three* required processes (Redis, Celery Worker, FastAPI Server) plus the frontend dev server.
*   Verification steps using API docs.
*   Updated How to Use section reflecting the current flow.
*   Included the project structure diagram.
*   Added troubleshooting tips, including CORS and history issues.
*   Mentioned the `run_history.json` file.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END