// Create this file: frontend/models/workflow.ts

// These interfaces should correspond to the Pydantic models defined in backend/app/models/workflow.py

export interface WorkflowConfig {
  // Fields that the user can configure via the API
  strategy_name?: string | null;
  seed?: number | null;
  start_date_str?: string | null;           // Expects YYYY-MM-DD format string
  validation_end_date?: string | null;    // Expects YYYY-MM-DD format string
  extend_simulation_end_date_str?: string | null; // Expects YYYY-MM-DD format string
  lookback_date_str?: string | null;      // Expects YYYY-MM-DD format string
  tickers?: string[] | null;            // List of stock tickers
  k_values?: number[] | null;           // List of k values for grid search (maps to K_VALUES_CANDIDATES)
  n_values?: number[] | null;           // List of n values for grid search (maps to N_VALUES_CANDIDATES)
  total_strategy_capital?: number | null; // Starting capital
  // Add other optional overrides if defined in backend Pydantic model (e.g., epochs, learning_rate)
}

export interface WorkflowRunResponse {
  // Response when starting a workflow
  message: string;
  task_id: string;
  run_id: string; // Often the same as task_id
}

export interface WorkflowStatusResponse {
  // Response from the status endpoint
  task_id: string;
  run_id: string;
  status: 'PENDING' | 'STARTED' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | 'RETRY' | 'REVOKED' | string; // Allow other Celery states as string
  details?: {
    message?: string;
    step?: string;      // Current step description
    progress?: number;  // Overall progress percentage (0-100)
    error?: string;     // Error message string on failure
    traceback?: string; // Optional traceback string on failure
    best_k?: number;    // Might be available in details on success/progress
    best_n?: number;
    val_loss?: number;  // Example: Include validation loss during training progress
    [key: string]: any; // Allow other arbitrary details from Celery meta
  } | null;
}

export interface FileInfo {
  // Relative paths to generated files within the run's output directory
  metrics_csv?: string | null;
  snapshot_csv?: string | null;
  report_txt?: string | null;
  training_history_plot?: string | null;
  full_period_plot?: string | null;
  equity_curve_plot?: string | null;
  test_period_plot?: string | null;
  log_file?: string | null;
}

export interface WorkflowResultResponse {
  // Response from the results endpoint for a completed run
  run_id: string;
  status: 'SUCCESS' | 'FAILURE'; // Final status
  message: string;
  best_k?: number | null;
  best_n?: number | null;
  output_dir_url: string; // Base URL to access files for this specific run
  files: FileInfo;        // Dictionary containing relative file paths
  error_details?: string | null; // Full traceback or detailed error on failure
}

// --- History Related Models ---
export interface RunHistoryItem {
     // Structure for a single item in the run history list
     run_id: string;
     start_time?: string | null; // ISO Format Date string (from backend JSON)
     end_time?: string | null;   // ISO Format Date string (from backend JSON)
     status: string;            // Celery status string (e.g., SUCCESS, FAILURE)
     best_k?: number | null;
     best_n?: number | null;
     message?: string | null;    // Short summary or error message
}

export interface RunHistoryResponse {
     // Response from the history endpoint
     history: RunHistoryItem[];
}

// --- CSV Data Structure Models ---
export interface MetricData {
    // Structure for data parsed from the metrics CSV
    Metric: string;                    // Name of the metric
    Strategy: string | number | null;  // Value for the strategy
    // Allow dynamic keys for Benchmark and Tracker columns
    [key: string]: string | number | null;
}

export interface SnapshotData {
    // Structure for data parsed from the snapshot CSV
    Date: string;                      // Date of the snapshot (ISO format string)
    Category: string;                  // e.g., 'Top Gainer', 'Current Holding'
    Rank?: number | string | null;     // Rank for movers (can be string 'N/A' or number)
    StockID: string;                   // Ticker symbol
    'Cumulative Return'?: number | string | null; // Parsed value (can be string 'N/A')
    'Snapshot Date Price'?: number | string | null;
    'Snapshot Date Daily Return'?: number | string | null;
    'Snapshot Date Score'?: number | string | null;
}