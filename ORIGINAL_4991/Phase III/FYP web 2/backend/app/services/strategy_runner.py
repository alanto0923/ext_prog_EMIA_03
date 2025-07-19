# backend/app/services/strategy_runner.py
import os
import json
import portalocker # Cross-platform file locking
import logging
from datetime import datetime, timezone, timedelta
from celery.result import AsyncResult
from typing import Dict, Any, Optional, List
import numpy as np
import re
import csv
import time

# --- CORRECTED/ADDED IMPORTS ---
# Import task by its registered name for clarity
from ..tasks import run_strategy_task # Task function itself
from ..celery_app import celery_app # Celery app instance
from ..core.config import settings # Configuration object
from ..core.strategy_logic import DEFAULT_CONFIG # Default parameters dict
# Import necessary models from the models module
from ..models.workflow import (
    WorkflowConfig,
    WorkflowRunResponse,      # <-- Was missing?
    WorkflowStatusResponse,   # <-- Was missing?
    WorkflowResultResponse,
    FileInfo,                 # <-- Was missing?
    RunHistoryItem,
    # RunHistoryResponse, # Keep if actually used, otherwise remove
    DefaultConfigResponse
)
# ------------------------------

logger = logging.getLogger(__name__)

# --- History File Persistence ---
HISTORY_FILE = os.path.join(settings.BASE_OUTPUT_DIR, "run_history.json") # <-- Uses settings
LOCK_FILE = os.path.join(settings.BASE_OUTPUT_DIR, "run_history.lock")
LOCK_TIMEOUT = 10 # seconds

def read_history() -> List[Dict[str, Any]]:
    # ... (read_history function remains the same) ...
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with portalocker.Lock(LOCK_FILE, mode='r', timeout=LOCK_TIMEOUT) as f_lock:
             with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                 content = f.read()
                 if not content:
                     return []
                 history = json.loads(content)
                 # Ensure it's a list, handle potential single object if file corrupt
                 return history if isinstance(history, list) else [history] if isinstance(history, dict) else []
    except portalocker.exceptions.LockException:
        logger.warning(f"Could not acquire lock on {LOCK_FILE} within {LOCK_TIMEOUT}s. Reading history might be stale or fail.")
        # Fallback: Try reading without lock (less safe)
        try:
             with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                  content = f.read()
                  if not content: return []
                  history = json.loads(content)
                  return history if isinstance(history, list) else [history] if isinstance(history, dict) else []
        except Exception as read_err:
             logger.error(f"Error reading history file {HISTORY_FILE} even without lock: {read_err}")
             return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {HISTORY_FILE}. Returning empty history.")
        return []
    except FileNotFoundError:
         return []
    except Exception as e:
         logger.error(f"Unexpected error reading history file {HISTORY_FILE}: {e}", exc_info=True)
         return []


def write_history(history_data: List[Dict[str, Any]]):
     # ... (write_history function remains the same) ...
    try:
        # Use 'w' mode within portalocker to overwrite
        with portalocker.Lock(LOCK_FILE, mode='w', timeout=LOCK_TIMEOUT) as f_lock:
            # Ensure parent directory exists before writing
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, default=str) # Use default=str for datetime
    except portalocker.exceptions.LockException:
         logger.error(f"Could not acquire lock on {LOCK_FILE} within {LOCK_TIMEOUT}s for writing history.")
    except Exception as e:
        logger.error(f"Error writing history file {HISTORY_FILE}: {e}", exc_info=True)


def sanitize_run_name(name: str) -> str:
    # ... (sanitize_run_name function remains the same) ...
    name = re.sub(r'[<>:"/\\|?*\s]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name[:50]

class StrategyRunnerService:

    def _update_history_file(self, run_data: RunHistoryItem):
        # ... (robust _update_history_file function remains the same) ...
        logger.debug(f"Attempting to update history file for run_id: {run_data.run_id}")
        history = read_history()
        updated = False
        run_data_dict = run_data.model_dump(exclude_none=True) # Use exclude_none
        if 'start_time' in run_data_dict and isinstance(run_data_dict['start_time'], datetime):
             run_data_dict['start_time'] = run_data_dict['start_time'].isoformat()
        if 'end_time' in run_data_dict and isinstance(run_data_dict['end_time'], datetime):
             run_data_dict['end_time'] = run_data_dict['end_time'].isoformat()

        for i, item in enumerate(history):
            # Ensure robust comparison even if run_id is missing in stored item
            if isinstance(item, dict) and item.get("run_id") == run_data.run_id:
                history[i].update(run_data_dict)
                updated = True
                logger.debug(f"Updated history entry for {run_data.run_id}")
                break

        if not updated:
            history.append(run_data_dict)
            logger.debug(f"Added new history entry for {run_data.run_id}")

        min_datetime_aware = datetime.min.replace(tzinfo=timezone.utc)
        def sort_key(item):
            start_time_val = item.get('start_time') if isinstance(item, dict) else None # Check if item is dict
            if isinstance(start_time_val, str) and start_time_val:
                try:
                    # Simplified parsing - rely on standard isoformat() output
                    return datetime.fromisoformat(start_time_val.replace('Z', '+00:00'))
                except Exception as e: # Catch broad errors during parsing attempt
                    logger.warning(f"Could not parse datetime '{start_time_val}' for sorting: {e}. Treating as oldest.")
                    return min_datetime_aware
            else:
                return min_datetime_aware

        try:
            # Filter out any non-dictionary items just in case before sorting
            valid_history = [item for item in history if isinstance(item, dict)]
            valid_history.sort(key=sort_key, reverse=True)
            write_history(valid_history) # Write back only valid items
        except Exception as sort_err:
            logger.error(f"History sorting or writing failed: {sort_err}", exc_info=True)
            # Attempt to write the unsorted list if sorting failed
            if not updated: # Avoid writing if update failed mid-way
                write_history(history)


    def start_workflow(self, config: WorkflowConfig) -> WorkflowRunResponse: # <-- Fixed model
        # ... (Configuration merging and Run ID generation logic remains the same) ...
        run_config = DEFAULT_CONFIG.copy() # <-- Fixed variable
        user_overrides = config.model_dump(exclude_unset=True)
        run_name = user_overrides.get('strategy_name') or run_config.get('STRATEGY_NAME', 'Unnamed_Run')
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") # Use UTC timestamp
        sanitized_name = sanitize_run_name(run_name)
        run_id = f"{timestamp}_{sanitized_name}"
        logger.info(f"Generated Run ID / Task ID: {run_id} (from name: '{run_name}')")
        run_output_dir_check = os.path.join(settings.BASE_OUTPUT_DIR, run_id) # <-- Uses settings
        count = 1
        original_run_id = run_id
        while os.path.exists(run_output_dir_check):
            run_id = f"{original_run_id}_{count}"
            run_output_dir_check = os.path.join(settings.BASE_OUTPUT_DIR, run_id)
            count += 1
            if count > 10:
                 logger.error(f"Run ID collision after {count} attempts for '{original_run_id}'. Aborting.")
                 raise FileExistsError(f"Could not create unique directory for run name '{run_name}'")
        if original_run_id != run_id:
            logger.warning(f"Run ID collision detected. Using modified ID: {run_id}")

        # ... (Internal Key Mapping and Config Cleaning logic remains the same) ...
        internal_key_map = {
             'strategy_name': 'STRATEGY_NAME', 'seed': 'SEED', 'start_date_str': 'START_DATE_STR',
             'validation_end_date': 'VALIDATION_END_DATE', 'extend_simulation_end_date_str': 'EXTEND_SIMULATION_END_DATE_STR',
             'lookback_date_str': 'LOOKBACK_DATE_STR', 'tickers': 'TICKERS', 'k_values': 'K_VALUES_CANDIDATES',
             'n_values': 'N_VALUES_CANDIDATES', 'total_strategy_capital': 'TOTAL_STRATEGY_CAPITAL',
        }
        for key, value in user_overrides.items():
             internal_key = internal_key_map.get(key)
             if internal_key and internal_key in run_config:
                 if key in ['tickers','k_values','n_values']:
                      if isinstance(value, list) and value: run_config[internal_key]=value; logger.info(f"User list for '{key}'.")
                      elif isinstance(value,list) and not value: logger.info(f"User empty list for '{key}'; using default.")
                      elif not isinstance(value, list): logger.warning(f"Expected list '{key}' got {type(value)}. Using default.")
                 elif value is not None: run_config[internal_key]=value
             elif key == 'strategy_name' and value is not None: run_config['STRATEGY_NAME']=value
             else:
                  if key != 'strategy_name': logger.warning(f"Unexpected config key {key}. Ignoring.")
        cleaned_config = {}; num_cleaned = 0
        for key, value in run_config.items():
            cleaned_value=value; type_changed=False
            if isinstance(value,np.ndarray): cleaned_value=value.tolist(); type_changed=True
            elif isinstance(value,(np.integer)): cleaned_value=int(value); type_changed=True
            elif isinstance(value,(np.floating)): cleaned_value=float(value) if np.isfinite(value) else None ; type_changed=True
            elif isinstance(value,(np.bool_)): cleaned_value=bool(value); type_changed=True
            elif isinstance(value,(np.void)): logger.warning(f"Skip numpy void key '{key}'."); continue
            elif isinstance(value, list):
                 needs_cleaning=any(isinstance(i,(np.ndarray,np.number,np.bool_)) for i in value)
                 if needs_cleaning:
                      logger.debug(f"Clean list key: {key}"); cleaned_list=[];
                      for item in value:
                           if isinstance(item,np.ndarray): cleaned_list.append(item.tolist())
                           elif isinstance(item,np.integer): cleaned_list.append(int(item))
                           elif isinstance(item,np.floating): cleaned_list.append(float(item) if np.isfinite(item) else None)
                           elif isinstance(item,np.bool_): cleaned_list.append(bool(item))
                           else: cleaned_list.append(item)
                      cleaned_value=cleaned_list; type_changed=True
                 else: cleaned_value=value
            cleaned_config[key]=cleaned_value;
            if type_changed: num_cleaned+=1
        if num_cleaned > 0: logger.info(f"Cleaned {num_cleaned} numpy types from config.")


        initial_history_record = RunHistoryItem(
            run_id=run_id,
            start_time=datetime.now(timezone.utc),
            status="PENDING",
            message="Task submitted to queue.",
        )
        self._update_history_file(initial_history_record)

        logger.info(f"Starting workflow run '{cleaned_config.get('STRATEGY_NAME')}' (Task/Run ID: {run_id})")

        task = run_strategy_task.apply_async( # <-- Fixed variable
            args=[run_id, cleaned_config],
            task_id=run_id
        )

        return WorkflowRunResponse(
             message="Workflow run started.",
             task_id=task.id,
             run_id=run_id
         )

    def get_workflow_status(self, run_id: str) -> WorkflowStatusResponse: # <-- Fixed model
        """Gets the status and details of a running task using the run_id."""
        task_result = AsyncResult(run_id, app=celery_app) # <-- Fixed variable
        status = task_result.status; details = None; task_info = task_result.info; task_backend_result = task_result.result
        if status == 'PENDING': details = {'message': 'Task is waiting to be processed.'}
        elif status == 'STARTED': details = task_info if isinstance(task_info, dict) else {'message': 'Task has started.'}
        elif status == 'PROGRESS': details = task_info if isinstance(task_info, dict) else {'message': 'Task is in progress.'}
        elif status == 'SUCCESS':
            details = {'message': 'Task completed successfully.'}
            # Safely access nested results
            res_dict = task_backend_result if isinstance(task_backend_result, dict) else {}
            details['best_k'] = res_dict.get('best_k'); details['best_n'] = res_dict.get('best_n')
        elif status == 'FAILURE':
            error_message = "Task failed."; error_details = None
            if isinstance(task_backend_result, Exception): error_message = f"Task failed: {str(task_backend_result)}"; error_details = task_result.traceback
            elif isinstance(task_backend_result, dict) and 'error' in task_backend_result: error_message = task_backend_result.get('message', error_message); error_details = task_backend_result.get('error')
            elif isinstance(task_info, dict) and 'error' in task_info: error_message = task_info.get('message', error_message); error_details = task_info.get('error')
            details = {'message': error_message, 'error': error_details if error_details else "No details available."}
        elif status == 'RETRY': details = task_info if isinstance(task_info, dict) else {'message': 'Task is being retried.'}
        elif status == 'REVOKED': details = {'message': 'Task was revoked.'}
        else: details = {'message': f'Task status: {status}'}
        return WorkflowStatusResponse( task_id=run_id, run_id=run_id, status=status, details=details )


    def get_workflow_results(self, run_id: str) -> Optional[WorkflowResultResponse]:
        """Retrieves the final results of a completed run using the run_id."""
        task_result = AsyncResult(run_id, app=celery_app) # <-- Fixed variable
        base_url = settings.OUTPUT_FILE_BASE_URL.rstrip('/'); run_output_url = f"{base_url}/{run_id}" # <-- Uses settings

        log_file_rel_path = f"{run_id}.log"
        log_file_abs_path = os.path.join(settings.BASE_OUTPUT_DIR, run_id, log_file_rel_path) # <-- Uses settings
        log_file_info = {"log_file": log_file_rel_path} if os.path.exists(log_file_abs_path) else {}


        if task_result.status == 'SUCCESS':
            result_data = task_result.result
            if isinstance(result_data, dict):
                files_dict = result_data.get("files", {}) if isinstance(result_data.get("files"), dict) else {}
                files_dict.update(log_file_info)
                return WorkflowResultResponse(run_id=run_id, status='SUCCESS', message=result_data.get("message", "Success"), best_k=result_data.get("best_k"), best_n=result_data.get("best_n"), output_dir_url=run_output_url, files=FileInfo(**files_dict), error_details=None) # <-- Uses FileInfo
            else:
                 logger.error(f"Task {run_id} succeeded but returned invalid result format: {type(result_data)}")
                 files_dict = {}
                 files_dict.update(log_file_info)
                 return WorkflowResultResponse(run_id=run_id, status='FAILURE', message="Internal Error: Invalid result format from successful task.", output_dir_url=run_output_url, files=FileInfo(**files_dict), error_details="Task returned non-dictionary result.") # <-- Uses FileInfo

        elif task_result.status == 'FAILURE':
            error_info=task_result.result; traceback_info=task_result.traceback; message=f"Workflow failed."; error_details=traceback_info; files_info={}
            result_data = task_result.info
            if isinstance(result_data, dict) and 'error' in result_data:
                message = result_data.get('message', message)
                error_details = result_data.get('error', traceback_info)
                files_info = result_data.get("files", {})
            elif isinstance(error_info, Exception): message = f"Workflow failed: {str(error_info)}"
            files_info.update(log_file_info)
            return WorkflowResultResponse(run_id=run_id, status='FAILURE', message=message, output_dir_url=run_output_url, files=FileInfo(**files_info), error_details=error_details) # <-- Uses FileInfo
        else: return None # Not finished or other state

    def get_default_config(self) -> Dict[str, Any]:
        """Returns the default configuration parameters."""
        configurable_defaults = {}; user_configurable_keys_model = list(WorkflowConfig.model_fields.keys());
        internal_key_map = {'strategy_name': None, 'seed': 'SEED', 'start_date_str': 'START_DATE_STR', 'validation_end_date': 'VALIDATION_END_DATE', 'extend_simulation_end_date_str': 'EXTEND_SIMULATION_END_DATE_STR', 'lookback_date_str': 'LOOKBACK_DATE_STR', 'tickers': 'TICKERS', 'k_values': 'K_VALUES_CANDIDATES', 'n_values': 'N_VALUES_CANDIDATES', 'total_strategy_capital': 'TOTAL_STRATEGY_CAPITAL'}
        for model_key in user_configurable_keys_model:
            internal_key = internal_key_map.get(model_key)
            if internal_key and internal_key in DEFAULT_CONFIG: # <-- Uses DEFAULT_CONFIG
                 configurable_defaults[model_key] = DEFAULT_CONFIG[internal_key] # <-- Uses DEFAULT_CONFIG
            elif model_key == 'strategy_name': configurable_defaults[model_key] = None
            else: logger.debug(f"No default found/mapped for key: {model_key}"); configurable_defaults[model_key] = None
        return configurable_defaults

    def get_history(self, limit: int = 50) -> List[RunHistoryItem]:
        """Retrieves a list of past workflow runs from the history file."""
        logger.info(f"Retrieving run history (limit: {limit}) from {HISTORY_FILE}")
        history_data = read_history() # Reads the raw dict list
        history_items: List[RunHistoryItem] = []
        for item_dict in history_data[:limit]:
             try:
                 if 'start_time' in item_dict and isinstance(item_dict['start_time'], str):
                      item_dict['start_time'] = datetime.fromisoformat(item_dict['start_time'].replace('Z', '+00:00'))
                 if 'end_time' in item_dict and isinstance(item_dict['end_time'], str):
                      item_dict['end_time'] = datetime.fromisoformat(item_dict['end_time'].replace('Z', '+00:00'))
                 history_items.append(RunHistoryItem(**item_dict))
             except Exception as parse_err:
                 logger.warning(f"Skipping history item due to parsing error: {parse_err}. Data: {item_dict}")
        logger.info(f"Retrieved {len(history_items)} history items.")
        return history_items


# Instantiate the service
strategy_runner_service = StrategyRunnerService()