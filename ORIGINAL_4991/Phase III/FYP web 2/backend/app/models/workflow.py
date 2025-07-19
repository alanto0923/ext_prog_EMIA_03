# backend/app/models/workflow.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class WorkflowConfig(BaseModel):
    # K_VALUES and N_VALUES REMOVED from here
    strategy_name: Optional[str] = Field(default=None, description="Custom name for this run")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    start_date_str: Optional[str] = Field(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$', description="Start date (YYYY-MM-DD)")
    validation_end_date: Optional[str] = Field(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$', description="End date for training/validation data (YYYY-MM-DD)")
    extend_simulation_end_date_str: Optional[str] = Field(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$', description="End date for extended simulation (YYYY-MM-DD)")
    lookback_date_str: Optional[str] = Field(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$', description="Date for snapshot analysis (YYYY-MM-DD)")
    tickers: Optional[List[str]] = Field(default=None, description="List of ticker symbols")
    # k_values: Optional[List[int]] = Field(default=None...) # REMOVED
    # n_values: Optional[List[int]] = Field(default=None...) # REMOVED
    total_strategy_capital: Optional[float] = Field(default=None, gt=0, description="Initial capital for simulation")

    model_config = { # Example remains the same, just won't include k/n
        "json_schema_extra": {
            "examples": [
                {
                    "strategy_name": "My HSI Run April (Defaults)",
                    "total_strategy_capital": 500000
                }
            ]
        }
    }

# ... rest of the models remain the same ...

class WorkflowRunResponse(BaseModel):
    message: str
    task_id: str
    run_id: str

class WorkflowStatusResponse(BaseModel):
    task_id: str
    run_id: str
    status: str
    details: Optional[Dict[str, Any]] = None

class FileInfo(BaseModel):
    metrics_csv: Optional[str] = None
    snapshot_csv: Optional[str] = None
    report_txt: Optional[str] = None
    training_history_plot: Optional[str] = None
    full_period_plot: Optional[str] = None
    equity_curve_plot: Optional[str] = None
    test_period_plot: Optional[str] = None
    log_file: Optional[str] = None

class WorkflowResultResponse(BaseModel):
    run_id: str
    status: str
    message: str
    best_k: Optional[int] = None # Still include best_k/n in RESULTS
    best_n: Optional[int] = None # Still include best_k/n in RESULTS
    output_dir_url: str
    files: FileInfo
    error_details: Optional[str] = None

class DefaultConfigResponse(BaseModel):
    # Ensure this returns relevant fields excluding k/n list inputs
    defaults: Dict[str, Any]

class RunHistoryItem(BaseModel):
     run_id: str
     start_time: Optional[datetime] = None
     end_time: Optional[datetime] = None
     status: str
     best_k: Optional[int] = None
     best_n: Optional[int] = None

class RunHistoryResponse(BaseModel):
     history: List[RunHistoryItem]

class RunHistoryItem(BaseModel):
     run_id: str
     start_time: Optional[datetime] = None # Often not easily available without DB
     end_time: Optional[datetime] = None
     status: str
     best_k: Optional[int] = None
     best_n: Optional[int] = None
     message: Optional[str] = None # Short summary or error message

class RunHistoryResponse(BaseModel):
     history: List[RunHistoryItem]
