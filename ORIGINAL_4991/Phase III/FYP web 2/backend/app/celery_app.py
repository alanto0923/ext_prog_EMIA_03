# backend/app/celery_app.py
from celery import Celery
from .core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks"]  # Point to the tasks module
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Add task tracking settings
    task_track_started=True,
    # task_send_sent_event=True, # Uncomment if you need sent events
    result_expires=3600 * 24 * 7, # Keep results for 7 days
)

if __name__ == "__main__":
    celery_app.start()