# backend/app/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from the backend directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    # Celery settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Output directory settings
    BASE_OUTPUT_DIR: str = "output" # Base directory relative to backend root
    OUTPUT_FILE_BASE_URL: str = "http://localhost:8000/static/output" # How frontend accesses files

    # Add other global settings if needed
    # Example: API_KEY: str | None = None

    class Config:
        # If using .env file
        env_file = '.env'
        env_file_encoding = 'utf-8'
        # Optional: If you prefer environment variables to have a prefix
        # env_prefix = 'APP_'

settings = Settings()

# Ensure the base output directory exists when the config is loaded
os.makedirs(settings.BASE_OUTPUT_DIR, exist_ok=True)