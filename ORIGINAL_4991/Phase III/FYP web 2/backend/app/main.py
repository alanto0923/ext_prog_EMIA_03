# backend/app/main.py
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .routers import workflow
from .core.config import settings

app = FastAPI(
    title="HSI DNN Strategy API",
    description="API to run and manage the HSI DNN Alpha Yield Strategy workflow.",
    version="0.1.0",
)

# --- CORS Middleware ---
# Allow requests from your frontend development server and production domain
origins = [
    "http://localhost:3000",  # Next.js default dev port
    # Add your production frontend URL here
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)


# --- API Routers ---
app.include_router(workflow.router)

# --- Static File Serving ---
# Mount the base 'output' directory to serve generated files
output_dir_path = os.path.abspath(settings.BASE_OUTPUT_DIR)
app.mount("/static/output", StaticFiles(directory=output_dir_path), name="output_files")


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the HSI DNN Strategy API"}

# --- Optional: Add lifespan events for startup/shutdown if needed ---
# from contextlib import asynccontextmanager
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Code to run on startup
#     print("API Starting up...")
#     yield
#     # Code to run on shutdown
#     print("API Shutting down...")
# app = FastAPI(lifespan=lifespan, ...) # Add lifespan to FastAPI constructor