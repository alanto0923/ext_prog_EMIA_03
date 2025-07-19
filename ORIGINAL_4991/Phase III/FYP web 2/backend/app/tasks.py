# backend/app/tasks.py
import os
import logging
from celery import shared_task
from celery.utils.log import get_task_logger
import time
import traceback # Import traceback

from .celery_app import celery_app
from .core.strategy_logic import run_full_strategy_workflow
from .core.config import settings

# Configure logging for the task runner
logger = get_task_logger(__name__) # Use Celery's task logger
logger.setLevel(logging.INFO) # Or DEBUG

# **** MODIFIED TASK SIGNATURE ****
@shared_task(bind=True, name="app.tasks.run_strategy_task") # Add explicit name
def run_strategy_task(self, run_id: str, config_overrides: dict): # Accepts run_id now
    """
    Celery task to run the full strategy workflow.
    'bind=True' gives access to the task instance (self).
    Accepts a pre-generated run_id used for output dir and logging.
    """
    task_id = self.request.id # Celery's internal task ID
    logger.info(f"Starting task {task_id} for Run ID: {run_id}")
    # Log if IDs differ (shouldn't happen with current service logic but good check)
    if task_id != run_id:
        logger.warning(f"Celery Task ID '{task_id}' differs from passed Run ID '{run_id}'. Using Run ID for outputs.")

    # Use the provided run_id for output directory
    run_output_dir = os.path.join(settings.BASE_OUTPUT_DIR, run_id)
    try: # Ensure directory can be created
        os.makedirs(run_output_dir, exist_ok=True)
    except OSError as e:
         logger.critical(f"!!! Cannot create output directory: {run_output_dir} - Error: {e} !!!", exc_info=True)
         # Update state with failure and raise exception
         failure_payload = {"message": f"Failed to create output directory: {e}", "run_id": run_id, "error": traceback.format_exc()}
         try: self.update_state(state='FAILURE', meta=failure_payload)
         except Exception as update_err: logger.error(f"Failed to update state on directory creation failure: {update_err}")
         raise # Re-raise the OSError

    # --- Configure File Logging for this specific run ---
    log_file_path = os.path.join(run_output_dir, f"{run_id}.log") # Use run_id for log name
    file_handler = logging.FileHandler(log_file_path, mode='w') # Overwrite log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO) # Set level for file handler

    # Get loggers that need file output and add handler
    celery_task_logger = get_task_logger(__name__)
    strategy_logic_logger = logging.getLogger("app.core.strategy_logic")

    # Prevent adding duplicate handlers on retries
    current_handlers = celery_task_logger.handlers + strategy_logic_logger.handlers
    handler_exists = any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == log_file_path for h in current_handlers)

    if not handler_exists:
        celery_task_logger.addHandler(file_handler)
        strategy_logic_logger.addHandler(file_handler)
        # Set levels if needed (Celery logger level set above)
        strategy_logic_logger.setLevel(logging.INFO)
        logger.info(f"Logging for Run ID {run_id} configured to: {log_file_path}")
    else:
        logger.info(f"File handler already exists for Run ID {run_id}: {log_file_path}")


    try:
        # Update status to STARTED, include run_id
        self.update_state(state='STARTED', meta={'step': 'Initializing', 'progress': 0, 'run_id': run_id})

        # Run the main workflow logic from strategy_logic.py
        result = run_full_strategy_workflow(
            config=config_overrides,
            output_base_dir=run_output_dir # Pass the directory path based on run_id
        )

        # Check the result structure from the workflow function
        if isinstance(result, dict) and result.get("message") == "Workflow completed successfully.":
            logger.info(f"Workflow task for Run ID {run_id} finished successfully.")
            # Result already contains run_id, best_k, best_n, files - Celery stores this
            return result
        else:
             logger.error(f"Workflow for Run ID {run_id} finished with unexpected result: {result}")
             raise ValueError("Workflow completed but returned invalid result structure.")

    except Exception as e:
        logger.critical(f"!!! Workflow Task for Run ID {run_id} FAILED Internally: {e} !!!", exc_info=True)
        # Prepare a failure payload including run_id and log file path
        failure_payload = {
            "message": f"Workflow failed: {str(e)}",
            "error": traceback.format_exc(), # Get full traceback string
            "run_id": run_id, # Include run_id in failure meta
            "output_dir": run_output_dir, # Might be useful for debugging access
            "files": { "log_file": f"{run_id}.log" } # Provide log file name
        }
        # Update state with failure info (Celery does this implicitly on raise, but this adds our custom payload)
        try:
             self.update_state(state='FAILURE', meta=failure_payload)
        except Exception as update_err:
             logger.error(f"Failed to update Celery state on task failure for Run ID {run_id}: {update_err}")

        # IMPORTANT: Re-raise the exception for Celery to correctly mark the task as FAILURE
        raise e
    finally:
        # --- Clean up logging handler for this task ---
        logger.info(f"Task for Run ID {run_id}: Cleaning up specific run log handlers.")
        # Remove handler safely
        if file_handler in celery_task_logger.handlers:
            celery_task_logger.removeHandler(file_handler)
        if file_handler in strategy_logic_logger.handlers:
            strategy_logic_logger.removeHandler(file_handler)
        file_handler.close()
        logger.info(f"Task for Run ID {run_id}: Log handler cleanup complete.")