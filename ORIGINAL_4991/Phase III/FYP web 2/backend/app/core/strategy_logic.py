# backend/app/core/strategy_logic.py
import pandas as pd
import numpy as np
# import ta # Import within functions that use it
import yfinance as yf
from datetime import datetime, timedelta,date 
import joblib
import os
import random
# Import tensorflow locally if needed, e.g., in train_dnn_model
# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Input, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting in backend/Celery
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import traceback
import logging
import statsmodels.api as sm
import time
from celery import current_task # Import Celery current_task

# --- Import NEW formatting utilities ---
# Ensure this file exists: backend/app/utils/formatting_utils.py
try:
    from ..utils.formatting_utils import format_metric, get_sorted_scores, dollar_formatter
except ImportError:
    logging.error("CRITICAL: formatting_utils.py not found or cannot be imported.")
    # Define dummy functions to prevent immediate crash, but things will look wrong.
    def format_metric(value, **kwargs): return str(value) if value is not None else "N/A"
    def get_sorted_scores(stock_set, score_dict, **kwargs): return [(s, score_dict.get(s, 'N/A')) for s in stock_set]
    def dollar_formatter(x, **kwargs): return str(x)

# --- Import report generation utility ---
# Ensure this file exists: backend/app/utils/report_utils.py
try:
    from ..utils.report_utils import generate_text_report
except ImportError:
     logging.error("CRITICAL: report_utils.py not found or cannot be imported.")
     def generate_text_report(report_data, output_file): # Dummy function
          logging.error("generate_text_report is unavailable.")
          pass

# --- Configuration Defaults ---
# These are the master defaults used if not overridden by user input via API
DEFAULT_CONFIG = {
    "STRATEGY_NAME": "DNN Alpha Yield Strategy (HSI)",
    "BASE_CURRENCY": "HKD",
    "SEED": 42,
    # Full HSI Ticker List (Ensure this is accurate or allow override)
    "TICKERS": ["0001.HK", "0002.HK", "0003.HK", "0005.HK", "0006.HK", "0011.HK", '0012.HK', '0016.HK', '0017.HK', '0027.HK', '0066.HK', '0101.HK', '0175.HK', '0241.HK', '0267.HK', '0285.HK', '0288.HK', '0291.HK', '0316.HK', '0322.HK', '0386.HK', '0388.HK', '0669.HK', '0688.HK', '0700.HK', '0762.HK', '0823.HK', '0836.HK', '0857.HK', '0868.HK', '0881.HK', '0883.HK', '0939.HK', '0941.HK', '0960.HK', '0968.HK', '0981.HK', '0992.HK', "1024.HK", '1038.HK', '1044.HK', '1088.HK', "1093.HK", '1099.HK', '1109.HK', '1113.HK', '1177.HK', '1209.HK', '1211.HK', '1299.HK', '1378.HK', '1398.HK', '1658.HK', '1810.HK', '1876.HK', '1928.HK', '1929.HK', '1997.HK', '2007.HK', '2015.HK', '2020.HK', '2269.HK', '2313.HK', '2318.HK', '2319.HK', '2331.HK', '2359.HK', '2382.HK', '2388.HK', '2628.HK', '2688.HK', '2899.HK', '3690.HK', '3692.HK', '3968.HK', '3988.HK', '6098.HK', '6618.HK', '6690.HK', '6862.HK', '9618.HK', '9633.HK', '9888.HK', '9901.HK', '9961.HK', '9988.HK', '9999.HK'],
    "BENCHMARK_TICKER": '^HSI',
    "TRACKER_FUND_TICKER": '2800.HK',
    "START_DATE_STR": "2020-03-24", # Default start date
    "SELECTED_ALPHA_COLS_PHASE_II": ['ATR_14d', 'MA_Crossover_10_50', 'Mean_Reversion_20d', 'Normalized_BBW_20d_2std', 'Stochastic_Oscillator_14d', 'Volume_Momentum_50d'],
    "FORWARD_RETURN_PERIOD": 1,
    "TRAIN_END_DATE": '2023-03-24', # Default train end
    "VALIDATION_END_DATE": '2024-03-24', # Default validation end
    "HIDDEN_UNITS": 10,
    "HIDDEN_ACTIVATION": 'relu',
    "OUTPUT_ACTIVATION": 'linear',
    "LEARNING_RATE": 0.00001,
    "EPOCHS": 200, # Keep reasonably high for potential quality
    "PATIENCE": 20,
    "BATCH_SIZE": 128,
    "DROPOUT_RATE": 0.3,
    "K_VALUES_CANDIDATES": list(range(50, 61)), # Fixed K range
    "N_VALUES_CANDIDATES": list(range(1, 11)),  # Fixed N range
    "VALIDATION_OPTIMIZATION_METRIC": 'Sharpe Ratio',
    "N_CANDIDATES_TO_TEST": 5,
    "RISK_FREE_RATE_ANNUAL": 0.0,
    "TRADING_DAYS_PER_YEAR": 252,
    "EXTEND_SIMULATION_END_DATE_STR": "2025-04-19", # Default sim end
    "TOTAL_STRATEGY_CAPITAL": 1000000, # Default capital
    "LOOKBACK_DATE_STR": "2025-04-16", # Default snapshot date
}

# --- Logging Setup ---
logger = logging.getLogger(__name__) # Get logger for this module

# --- Reproducibility Setup ---
def set_seeds(seed_value):
    """Sets random seeds for Python, NumPy, and TensorFlow."""
    logger.info(f"--- Setting Random Seeds to {seed_value} ---")
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        # Import TF only when needed
        import tensorflow as tf
        tf.random.set_seed(seed_value)
        # Optional: Configure GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"TensorFlow GPU memory growth enabled for {len(gpus)} GPU(s).")
            except RuntimeError as e:
                # Only log error if memory growth failed, don't crash
                logger.error(f"Could not set GPU memory growth: {e}")
        else: logger.info("No GPUs detected by TensorFlow.")
    except ImportError:
        logger.warning("TensorFlow not found. Skipping TF random seed setting.")
    except Exception as e:
         logger.error(f"Error during TensorFlow seed/GPU setup: {e}", exc_info=True)


# --- Helper function ---
def _check_columns(df):
    """Checks if DataFrame has required OHLCV columns."""
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(df.columns):
        missing_cols = required_cols - set(df.columns)
        # Log clearly which columns are missing
        logger.error(f"Input DataFrame missing required columns: {missing_cols}. Available: {list(df.columns)}")
        raise ValueError(f"Input DataFrame missing required OHLCV columns: {missing_cols}.")

# === SECTION 1: Alpha Factor Calculation Functions ===
def calculate_price_momentum_10d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    price_col = 'Close'
    close_delayed = df[price_col].shift(10)
    close_delayed_safe = close_delayed.replace(0, np.nan)
    momentum = (df[price_col] - close_delayed_safe) / close_delayed_safe
    return momentum.replace([np.inf, -np.inf], np.nan) * 100

def calculate_ma_crossover_10_50(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    price_col = 'Close'
    sma10 = df[price_col].rolling(window=10, min_periods=10).mean()
    sma50 = df[price_col].rolling(window=50, min_periods=50).mean()
    return sma10 - sma50

def calculate_volume_momentum_50d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    volume_numeric = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
    return (volume_numeric - volume_numeric.shift(50)).fillna(0)

def calculate_roc_50d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    price_col = 'Close'
    close_delayed = df[price_col].shift(50)
    close_delayed_safe = close_delayed.replace(0, np.nan)
    roc = ((df[price_col] - close_delayed_safe) / close_delayed_safe) * 100
    return roc.replace([np.inf, -np.inf], np.nan)

def calculate_mean_reversion_20d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    price_col = 'Close'
    sma20 = df[price_col].rolling(window=20, min_periods=20).mean()
    price_numeric = pd.to_numeric(df[price_col], errors='coerce')
    return sma20 - price_numeric

def calculate_stochastic_oscillator_14d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    price_col = 'Close'
    low_14 = df['Low'].rolling(window=14, min_periods=14).min()
    high_14 = df['High'].rolling(window=14, min_periods=14).max()
    denominator = high_14 - low_14
    denominator_safe = denominator.replace(0, np.nan)
    price_numeric = pd.to_numeric(df[price_col], errors='coerce')
    stoch_k = 100 * (price_numeric - low_14) / denominator_safe
    return stoch_k.replace([np.inf, -np.inf], np.nan)

def calculate_atr_14d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    try:
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        close = df['Close'].astype(float)
        import ta # Import locally
        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14, fillna=False
        ).average_true_range()
    except ImportError:
         logger.error("Technical Analysis library 'ta' not installed. Cannot calculate ATR.")
         atr = pd.Series(np.nan, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating ATR with ta library: {e}", exc_info=False)
        atr = pd.Series(np.nan, index=df.index)
    return atr

def calculate_daily_high_low_range(df: pd.DataFrame) -> pd.Series:
     _check_columns(df)
     high_numeric = pd.to_numeric(df['High'], errors='coerce')
     low_numeric = pd.to_numeric(df['Low'], errors='coerce')
     return high_numeric - low_numeric

def calculate_normalized_bollinger_band_width_20d_2std(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    price_col = 'Close'
    price_numeric = pd.to_numeric(df[price_col], errors='coerce')
    sma20 = price_numeric.rolling(window=20, min_periods=20).mean()
    std20 = price_numeric.rolling(window=20, min_periods=20).std(ddof=1)
    sma20_safe = sma20.replace(0, np.nan)
    bbw_normalized = (4 * std20) / sma20_safe
    return bbw_normalized.replace([np.inf, -np.inf], np.nan)

def calculate_volume_roc_10d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    volume_numeric = pd.to_numeric(df['Volume'], errors='coerce')
    vol_delayed = volume_numeric.shift(10)
    vol_delayed_safe = vol_delayed.replace(0, np.nan)
    vroc = ((volume_numeric / vol_delayed_safe) - 1) * 100
    return vroc.replace([np.inf, -np.inf], np.nan)

def calculate_sma_20d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    price_col = 'Close'
    price_numeric = pd.to_numeric(df[price_col], errors='coerce')
    return price_numeric.rolling(window=20, min_periods=20).mean()

def calculate_ema_20d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    price_col = 'Close'
    price_numeric = pd.to_numeric(df[price_col], errors='coerce')
    return price_numeric.ewm(span=20, adjust=False, min_periods=20).mean()

def calculate_rsi_14d(df: pd.DataFrame) -> pd.Series:
    _check_columns(df)
    price_col = 'Close'
    try:
        import ta # Import locally
        close_float = df[price_col].astype(float)
        rsi = ta.momentum.RSIIndicator(
            close=close_float, window=14, fillna=False
        ).rsi()
    except ImportError:
         logger.error("Technical Analysis library 'ta' not installed. Cannot calculate RSI.")
         rsi = pd.Series(np.nan, index=df.index)
    except Exception as e:
        logger.warning(f"Error calculating RSI with ta library: {e}", exc_info=False)
        rsi = pd.Series(np.nan, index=df.index)
    return rsi

def calculate_all_alpha_factors(df: pd.DataFrame, feature_names_list: list) -> pd.DataFrame:
    """Calculates all specified alpha factors for the input DataFrame."""
    df_out = df.copy()
    ticker_id = df_out['stock_id'].iloc[0] if 'stock_id' in df_out.columns and not df_out.empty else 'dataframe'
    logger.debug(f"Calculating alphas for {ticker_id}...")
    # Define map inside or globally if preferred
    alpha_calculators = {
        'ATR_14d': calculate_atr_14d,
        'MA_Crossover_10_50': calculate_ma_crossover_10_50,
        'Mean_Reversion_20d': calculate_mean_reversion_20d,
        'Normalized_BBW_20d_2std': calculate_normalized_bollinger_band_width_20d_2std,
        'Stochastic_Oscillator_14d': calculate_stochastic_oscillator_14d,
        'Volume_Momentum_50d': calculate_volume_momentum_50d,
        # Add others based on your default config or passed list
        'Price_Momentum_10d': calculate_price_momentum_10d,
        'ROC_50d': calculate_roc_50d,
        'Daily_High_Low_Range': calculate_daily_high_low_range,
        'VROC_10d': calculate_volume_roc_10d,
        'SMA_20d': calculate_sma_20d,
        'EMA_20d': calculate_ema_20d,
        'RSI_14d': calculate_rsi_14d,
    }
    for factor_name in feature_names_list:
        if factor_name in alpha_calculators:
            try: df_out[factor_name] = alpha_calculators[factor_name](df_out)
            except Exception as e:
                logger.warning(f"Error calculating alpha '{factor_name}' for {ticker_id}: {e}")
                df_out[factor_name] = np.nan # Ensure column exists even if calc fails
        elif factor_name == 'Trading_Volume': # Handle alias if needed
             if 'Volume' in df_out.columns and 'Trading_Volume' not in df_out.columns:
                  df_out.rename(columns={'Volume': 'Trading_Volume'}, inplace=True)
             elif 'Trading_Volume' not in df_out.columns: df_out['Trading_Volume'] = np.nan # Add as NaN if source Volume is missing
        elif factor_name not in df_out.columns: # Check if factor might already exist (e.g., from yfinance)
             logger.warning(f"No calculator or column found for requested factor: {factor_name}. Adding as NaN.")
             df_out[factor_name] = np.nan
    return df_out
# === END SECTION 1 ===


# === SECTION 2: Data Fetching and Initial Processing ===
def fetch_and_process_raw_data(tickers, start_date_str, end_date_str, benchmark_ticker, selected_factors, output_csv_path):
    """Fetches raw data, calculates alphas, filters, cleans, and saves."""
    logger.info(f"Starting raw data fetch & processing: {start_date_str} to {end_date_str}")
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError as date_err:
        logger.error(f"Invalid date format provided: {date_err}")
        raise ValueError(f"Invalid start or end date format: {date_err}")

    yf_end_date = end_date + timedelta(days=1) # yfinance excludes end date
    processed_data_list = []
    fetch_errors = {}

    # Fetch Benchmark (optional but good practice)
    try:
        logger.info(f"Fetching benchmark data: {benchmark_ticker}")
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=yf_end_date, auto_adjust=True, progress=False, ignore_tz=True, timeout=30)
        if benchmark_data.empty: logger.warning(f"No benchmark data found for {benchmark_ticker}.")
        else: logger.info(f"Benchmark data fetched. Shape: {benchmark_data.shape}")
    except Exception as e:
        logger.error(f"Failed to download benchmark {benchmark_ticker}: {e}")
        fetch_errors[benchmark_ticker] = str(e)

    # Fetch Tickers
    num_tickers = len(tickers)
    for i, ticker in enumerate(tickers):
        if (i + 1) % 10 == 0 or i == 0 or i == num_tickers - 1:
            logger.info(f"--- Processing ticker {i+1}/{num_tickers}: {ticker} ---")
        try:
            data = None # Initialize data
            # Add simple retry mechanism for yfinance download
            for attempt in range(2): # Try up to 2 times
                try:
                    data = yf.download(ticker, start=start_date, end=yf_end_date, auto_adjust=True, progress=False, ignore_tz=True, timeout=20)
                    if data is not None and not data.empty: # Check data is not None
                         break # Success, exit retry loop
                    else:
                         logger.warning(f"No data found for {ticker} on attempt {attempt + 1}.")
                         if attempt == 1: # Last attempt failed
                             fetch_errors[ticker] = "No data returned by yfinance after retries"
                except Exception as download_err:
                     logger.warning(f"yfinance download error for {ticker} (attempt {attempt+1}): {download_err}")
                     if attempt == 1: # Last attempt failed
                         fetch_errors[ticker] = f"Download failed after retries: {download_err}"
                     time.sleep(1) # Wait before retrying

            if ticker in fetch_errors or data is None or data.empty:
                 logger.warning(f"Skipping ticker {ticker} due to fetch error or no data.")
                 continue # Skip to next ticker

            # Process downloaded data
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data['stock_id'] = ticker
            alpha_data = calculate_all_alpha_factors(data.copy(), selected_factors)

            if 'Close' in alpha_data.columns: alpha_data['price'] = alpha_data['Close']
            else: alpha_data['price'] = np.nan

            alpha_data.reset_index(inplace=True)
            # Standardize Date column name
            if 'Datetime' in alpha_data.columns: alpha_data.rename(columns={'Datetime': 'Date'}, inplace=True)
            elif 'index' in alpha_data.columns and 'Date' not in alpha_data.columns: alpha_data.rename(columns={'index': 'Date'}, inplace=True)

            if 'Date' not in alpha_data.columns or not pd.api.types.is_datetime64_any_dtype(alpha_data['Date']):
                logger.warning(f"Invalid or missing 'Date' column for {ticker} after processing. Skipping.")
                fetch_errors[ticker] = "Missing/Invalid Date Column post-processing"
                continue

            alpha_data['Date'] = pd.to_datetime(alpha_data['Date']).dt.normalize()
            processed_data_list.append(alpha_data)

        except ValueError as ve: # Catch missing OHLCV columns from _check_columns
            logger.error(f"ValueError processing {ticker}: {ve}")
            fetch_errors[ticker] = str(ve)
        except Exception as e:
            logger.error(f"Unexpected error processing {ticker}: {e}", exc_info=True)
            fetch_errors[ticker] = f"Unexpected Error: {e}"

    if fetch_errors:
         logger.warning(f"Completed fetch/process loop. Encountered errors for {len(fetch_errors)} tickers.")
         # Log specific errors if needed: logger.debug(fetch_errors)
    if not processed_data_list:
        raise RuntimeError("No ticker data was processed successfully. Workflow cannot proceed.")

    logger.info("Combining all successfully processed ticker data...")
    combined_df_all_alphas = pd.concat(processed_data_list, ignore_index=True)
    combined_df_all_alphas['Date'] = pd.to_datetime(combined_df_all_alphas['Date'])
    combined_df_all_alphas.sort_values(by=['stock_id', 'Date'], inplace=True)
    logger.info(f"Combined raw data shape (successful tickers): {combined_df_all_alphas.shape}")

    logger.info("Filtering combined data for essential columns and selected factors...")
    BASE_COLS_TO_KEEP = ['Date', 'stock_id', 'price', 'Open', 'High', 'Low', 'Close', 'Volume'] # Keep OHLCV for future use/checks
    FINAL_COLS_TO_KEEP = sorted(list(set(BASE_COLS_TO_KEEP + selected_factors)))
    available_columns = combined_df_all_alphas.columns.tolist()
    actual_cols_to_keep = [col for col in FINAL_COLS_TO_KEEP if col in available_columns]
    missing_cols = set(FINAL_COLS_TO_KEEP) - set(actual_cols_to_keep)
    if missing_cols:
        logger.warning(f"Selected/Base columns NOT FOUND in combined data: {missing_cols}. These factors won't be available.")

    # Ensure required factors for the model are present before filtering
    required_model_factors = set(selected_factors)
    available_factors = set(actual_cols_to_keep)
    missing_model_factors = required_model_factors - available_factors
    if missing_model_factors:
         raise ValueError(f"Cannot proceed: Required model factors are missing from the combined data: {missing_model_factors}")

    final_df_filtered = combined_df_all_alphas[actual_cols_to_keep].copy()
    logger.info(f"Filtered DataFrame shape: {final_df_filtered.shape}")
    logger.info(f"Columns kept: {final_df_filtered.columns.tolist()}")

    logger.info("Handling potential NaNs in selected alpha factors...")
    alpha_columns_in_filtered = [col for col in selected_factors if col in final_df_filtered.columns]
    initial_nan_count = final_df_filtered[alpha_columns_in_filtered].isnull().sum().sum()
    if initial_nan_count > 0:
        logger.info(f"Found {initial_nan_count} NaNs in alpha factors. Applying forward fill then backward fill per stock...")
        # Use apply with lambda for group-wise ffill/bfill robustness
        final_df_filtered[alpha_columns_in_filtered] = final_df_filtered.groupby('stock_id', group_keys=False)[alpha_columns_in_filtered].apply(lambda x: x.ffill().bfill())
        final_nan_count = final_df_filtered[alpha_columns_in_filtered].isnull().sum().sum()
        logger.info(f"NaN count in alpha factors after ffill/bfill: {final_nan_count}")
        if final_nan_count > 0:
             logger.warning(f"{final_nan_count} NaNs remain after ffill/bfill. Dropping rows with any NaNs in alpha factors.")
             final_df_filtered.dropna(subset=alpha_columns_in_filtered, inplace=True)

    # Final check for NaN price and empty dataframe
    initial_rows = len(final_df_filtered)
    final_df_filtered.dropna(subset=['price'], inplace=True)
    rows_dropped_price = initial_rows - len(final_df_filtered)
    if rows_dropped_price > 0: logger.info(f"Dropped {rows_dropped_price} rows with NaN price.")

    final_df_filtered.reset_index(drop=True, inplace=True)
    logger.info(f"Final cleaned DataFrame shape for saving: {final_df_filtered.shape}")
    if final_df_filtered.empty:
        raise RuntimeError("Filtered DataFrame is empty after cleaning. Cannot proceed.")

    logger.info(f"Saving final filtered dataframe to {output_csv_path}...")
    final_df_filtered.to_csv(output_csv_path, index=False)
    logger.info("Raw data processing complete.")
    return output_csv_path
# === END SECTION 2 ===


# === SECTION 3: Data Preprocessing for DNN ===
def preprocess_data_for_dnn(input_csv_path, output_dir, alpha_cols_list, forward_return_period, train_end_dt_str, val_end_dt_str):
    """Loads data, calculates target, splits, scales, and saves processed data."""
    logger.info("--- Starting DNN Data Preprocessing ---")
    try:
        # Load data
        data_full = pd.read_csv(input_csv_path, parse_dates=['Date'])
        logger.info(f"Loaded filtered data from {input_csv_path}. Shape: {data_full.shape}")
        data_full.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check for required columns
        required_cols = set(alpha_cols_list + ['Date', 'stock_id', 'price'])
        missing_cols = required_cols - set(data_full.columns)
        if missing_cols:
            raise ValueError(f"Input CSV '{input_csv_path}' is missing required columns: {missing_cols}")

        # Handle potential NaNs before return calculation
        check_cols_nan = alpha_cols_list + ['price']
        initial_nan_count = data_full[check_cols_nan].isnull().sum().sum()
        if initial_nan_count > 0:
            logger.warning(f"Found {initial_nan_count} NaN values in features/price before return calc. Applying ffill/bfill per stock.")
            # Group apply is generally safer for ffill/bfill
            data_full[alpha_cols_list] = data_full.groupby('stock_id', group_keys=False)[alpha_cols_list].apply(lambda x: x.ffill().bfill())
            # Drop rows where price is still NaN after potential fill (shouldn't happen if handled in previous step)
            data_full.dropna(subset=['price'], inplace=True)
            # Drop rows where factors are still NaN after ffill/bfill
            rows_before_factor_drop = len(data_full)
            data_full.dropna(subset=alpha_cols_list, inplace=True)
            if len(data_full) < rows_before_factor_drop:
                 logger.warning(f"Dropped {rows_before_factor_drop - len(data_full)} rows with remaining NaNs in factors.")

            if data_full.empty:
                raise ValueError("DataFrame empty after handling initial NaNs.")
            logger.info(f"Shape after initial NaN handling: {data_full.shape}")


        # Calculate forward returns
        logger.info(f"Calculating {forward_return_period}-day forward returns (target variable)...")
        data_full = data_full.sort_values(by=['stock_id', 'Date']) # Ensure correct order for shift
        price_t_plus_n = data_full.groupby('stock_id')['price'].shift(-forward_return_period)
        data_full['forward_return'] = np.where(
            data_full['price'] > 1e-9, # Avoid division by zero or small numbers
            (price_t_plus_n / data_full['price']) - 1,
            np.nan
        )

        data_full.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle potential infs from division
        rows_before_target_drop = len(data_full)
        data_full.dropna(subset=['forward_return'], inplace=True) # Drop rows where target cannot be calculated
        logger.info(f"Dropped {rows_before_target_drop - len(data_full)} rows with NaN forward returns.")
        logger.info(f"Shape after calculating and cleaning targets: {data_full.shape}")

        if data_full.empty:
            raise ValueError("DataFrame empty after calculating forward returns. Check data range and forward period.")

        # Time-based split
        TARGET_COL = 'forward_return'
        logger.info("Performing time-based split...")
        data_full = data_full.sort_values(by='Date') # Ensure sorting by date for splitting
        try:
            train_end_dt_obj = pd.to_datetime(train_end_dt_str)
            val_end_dt_obj = pd.to_datetime(val_end_dt_str)
        except ValueError as date_err:
             raise ValueError(f"Invalid date format in configuration: {date_err}")

        train_data = data_full[data_full['Date'] <= train_end_dt_obj].copy()
        validation_data = data_full[(data_full['Date'] > train_end_dt_obj) & (data_full['Date'] <= val_end_dt_obj)].copy()
        test_data = data_full[data_full['Date'] > val_end_dt_obj].copy()

        logger.info(f"Data split shapes: Train={train_data.shape}, Validation={validation_data.shape}, Test={test_data.shape}")

        # Check if splits resulted in empty dataframes (critical)
        if train_data.empty:
            raise ValueError(f"Train data split is empty. Check START_DATE_STR ('{data_full['Date'].min().date()}') and TRAIN_END_DATE ('{train_end_dt_str}').")
        if validation_data.empty:
             raise ValueError(f"Validation data split is empty. Check TRAIN_END_DATE ('{train_end_dt_str}') and VALIDATION_END_DATE ('{val_end_dt_str}').")
        if test_data.empty:
            logger.warning(f"Test data split is empty. This is expected if VALIDATION_END_DATE ('{val_end_dt_str}') is the last date in the input data ('{data_full['Date'].max().date()}').")

        # Final NaN check on feature columns within each split
        if train_data[alpha_cols_list].isnull().sum().sum() > 0: raise ValueError("NaNs found in final train features!")
        if validation_data[alpha_cols_list].isnull().sum().sum() > 0: raise ValueError("NaNs found in final validation features!")
        if not test_data.empty and test_data[alpha_cols_list].isnull().sum().sum() > 0: raise ValueError("NaNs found in final test features!")

        # Prepare feature matrices (X) and target vectors (y)
        X_train = train_data[alpha_cols_list].astype(np.float32)
        y_train = train_data[TARGET_COL].values.astype(np.float32)
        X_val = validation_data[alpha_cols_list].astype(np.float32)
        y_val = validation_data[TARGET_COL].values.astype(np.float32)

        # Handle potentially empty test data
        if not test_data.empty:
            X_test = test_data[alpha_cols_list].astype(np.float32)
            y_test = test_data[TARGET_COL].values.astype(np.float32)
            test_context = test_data[['Date', 'stock_id']].reset_index(drop=True)
        else:
            n_features = len(alpha_cols_list)
            X_test = np.empty((0, n_features), dtype=np.float32)
            y_test = np.empty((0,), dtype=np.float32)
            test_context = pd.DataFrame(columns=['Date', 'stock_id'])

        # Save context data (Date, stock_id) for linking predictions back later
        logger.info("Saving context data (Date, stock_id) for each split...")
        train_context = train_data[['Date', 'stock_id']].reset_index(drop=True)
        val_context = validation_data[['Date', 'stock_id']].reset_index(drop=True)
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        train_context.to_pickle(os.path.join(output_dir, 'train_context.pkl'))
        val_context.to_pickle(os.path.join(output_dir, 'val_context.pkl'))
        test_context.to_pickle(os.path.join(output_dir, 'test_context.pkl'))

        # Scale features using MinMaxScaler (fitting on Train only)
        logger.info("Scaling features using MinMaxScaler (fitting on Train data only)...")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val) # Apply same scaling to validation
        # Scale test set only if it has data
        if X_test.shape[0] > 0:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = np.empty((0, X_train_scaled.shape[1]), dtype=np.float32) # Empty scaled array

        # Check for NaNs AFTER scaling (can happen with zero-variance features)
        nan_check_scaled = {
             'train': np.isnan(X_train_scaled).sum(),
             'val': np.isnan(X_val_scaled).sum(),
             'test': np.isnan(X_test_scaled).sum() if X_test_scaled.shape[0] > 0 else 0
        }
        total_nan_scaled = sum(nan_check_scaled.values())
        if total_nan_scaled > 0:
            logger.error(f"NaNs found AFTER scaling! Counts: {nan_check_scaled}")
            raise ValueError("NaNs detected after scaling features. Check for constant/zero-variance features.")
        logger.info("No NaNs found after scaling.")

        # Save processed numpy arrays and the scaler
        logger.info(f"Saving processed numpy arrays and scaler to '{output_dir}'...")
        np.save(os.path.join(output_dir, 'X_train_scaled.npy'), X_train_scaled)
        np.save(os.path.join(output_dir, 'y_train_continuous.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_val_scaled.npy'), X_val_scaled)
        np.save(os.path.join(output_dir, 'y_val_continuous.npy'), y_val)
        np.save(os.path.join(output_dir, 'X_test_scaled.npy'), X_test_scaled) # Save potentially empty test arrays
        np.save(os.path.join(output_dir, 'y_test_continuous.npy'), y_test)

        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)

        feature_list_path = os.path.join(output_dir, 'feature_list.txt')
        with open(feature_list_path, 'w') as f:
            for feature in alpha_cols_list:
                f.write(f"{feature}\n")

        logger.info("DNN data preprocessing complete.")
        return scaler_path, feature_list_path

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_csv_path}")
        raise
    except ValueError as ve:
        logger.error(f"Data processing error: {ve}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during DNN preprocessing: {e}", exc_info=True)
        raise
# === END SECTION 3 ===


# === SECTION 4: DNN Model Training ===
# Note: Import TF locally inside train_dnn_model if not already imported globally
from tensorflow.keras.models import Sequential, load_model # Keep these imports
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback

class EpochLogger(Callback):
    """Logs epoch progress (Celery update commented out)."""
    def __init__(self, total_epochs, major_step_num, total_major_steps): # Pass progress info
        super().__init__()
        self.total_epochs = total_epochs
        self.major_step_num = major_step_num # Store for potential progress calculation
        self.total_major_steps = total_major_steps # Store for potential progress calculation
        # Check if running inside a Celery task context
        self.task_context = None
        if current_task and hasattr(current_task, 'request') and current_task.request:
             self.task_context = current_task # Store task instance
        else:
             logger.debug("EpochLogger initialized outside of Celery task context.")

        # Log roughly every 5% of epochs or every 10 epochs, whichever is more frequent, but at least once per 50
        self.epochs_per_log = max(1, min(10, total_epochs // 20)) if total_epochs > 0 else 1

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % self.epochs_per_log == 0 :
             logger.info(f"--- Starting Epoch {epoch+1}/{self.total_epochs} ---")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_str = " - ".join([f"{k}: {logs[k]:.5f}" for k in sorted(logs.keys())])
        stopped_training = getattr(self.model, 'stop_training', False)

        # Log every N epochs or on the last epoch or if stopped
        if (epoch + 1) % self.epochs_per_log == 0 or epoch == self.total_epochs - 1 or stopped_training:
            logger.info(f"Epoch {epoch+1}/{self.total_epochs} completed - {log_str}")

        # --- Celery Progress Update Section (Now Active) ---
        if self.task_context and ((epoch + 1) % self.epochs_per_log == 0 or epoch == self.total_epochs - 1 or stopped_training):
            try:
                # Calculate progress within this major step
                step_progress = int(100 * (epoch + 1) / self.total_epochs) if self.total_epochs > 0 else 100
                # Map step progress to overall progress (Training as step 3/6)
                step_start_progress = int(100 * (self.major_step_num - 1) / self.total_major_steps)
                step_end_progress = int(100 * self.major_step_num / self.total_major_steps)
                overall_progress = step_start_progress + int(step_progress * (step_end_progress - step_start_progress) / 100)

                # Prepare metadata for Celery state
                meta = {
                    'step': f'Training DNN (Epoch {epoch+1}/{self.total_epochs})',
                    'progress': min(overall_progress, 100), # Cap progress at 100
                    'val_loss': logs.get('val_loss') # Include latest validation loss if available
                }
                self.task_context.update_state(state='PROGRESS', meta=meta)
                logger.debug(f"Celery task state updated: Progress {overall_progress}%")
            except Exception as e:
                # Log warning but don't crash the training
                logger.warning(f"Could not update Celery task state during training: {e}")
        # --- End Celery Update Section ---
        pass # End of on_epoch_end

def train_dnn_model(processed_data_dir, model_save_path, feature_list_path, plot_save_path,
                    hidden_units, hidden_activation, output_activation, learning_rate,
                    epochs, patience, batch_size, dropout_rate, seed,
                    major_step_num, total_major_steps): # Added progress args
    """Trains the DNN model and saves it."""
    logger.info("--- Starting DNN Model Training ---")
    set_seeds(seed) # Ensure seeds are set for this step
    try:
        logger.info(f"Loading preprocessed data from '{processed_data_dir}'...")
        X_train_scaled=np.load(os.path.join(processed_data_dir,'X_train_scaled.npy'))
        y_train=np.load(os.path.join(processed_data_dir,'y_train_continuous.npy'))
        X_val_scaled=np.load(os.path.join(processed_data_dir,'X_val_scaled.npy'))
        y_val=np.load(os.path.join(processed_data_dir,'y_val_continuous.npy'))
        X_test_scaled=np.load(os.path.join(processed_data_dir,'X_test_scaled.npy'))
        y_test=np.load(os.path.join(processed_data_dir,'y_test_continuous.npy'))

        with open(feature_list_path, 'r') as f: features = [line.strip() for line in f if line.strip()]; n_features = len(features)
        logger.info(f"Data loaded. Features ({n_features}): {features[:5]}... Shapes: Tr={X_train_scaled.shape}, V={X_val_scaled.shape}, Te={X_test_scaled.shape}")

        # --- Input Data Validation ---
        if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any() or np.isnan(y_train).any() or np.isinf(y_train).any(): raise ValueError("NaNs/Infs in training data!")
        if np.isnan(X_val_scaled).any() or np.isinf(X_val_scaled).any() or np.isnan(y_val).any() or np.isinf(y_val).any(): raise ValueError("NaNs/Infs in validation data!")
        is_test_data_valid = False
        if X_test_scaled.shape[0]>0 and y_test.shape[0]>0 and X_test_scaled.shape[0]==y_test.shape[0]:
             if np.isnan(X_test_scaled).any() or np.isinf(X_test_scaled).any() or np.isnan(y_test).any() or np.isinf(y_test).any(): logger.warning("NaNs/Infs in test data! Skipping test evaluation.")
             else: is_test_data_valid = True
        else: logger.info("Test data empty or shape mismatch. Skipping test evaluation.")

        # --- Model Definition ---
        logger.info("Defining the DNN model...")
        import tensorflow as tf # Local import
        # Clear session? Optional, can help with memory in long-running workers
        # tf.keras.backend.clear_session()
        model = tf.keras.models.Sequential(name="DNN_Alpha_Yield_Predictor")
        model.add(tf.keras.layers.Input(shape=(n_features,), name="Input_Alphas"))
        model.add(tf.keras.layers.Dense(hidden_units, activation=hidden_activation, name="Hidden_Layer_ReLU"))
        if 0 < dropout_rate < 1: model.add(tf.keras.layers.Dropout(dropout_rate, name="Dropout_Layer", seed=seed))
        elif dropout_rate > 0: logger.warning(f"Invalid dropout rate {dropout_rate}. Disabling dropout.")
        model.add(tf.keras.layers.Dense(1, activation=output_activation, name="Output_Yield_Prediction"))
        summary_list = []; model.summary(print_fn=lambda x: summary_list.append(x)); logger.info("Model Summary:\n" + "\n".join(summary_list))

        # --- Model Compilation ---
        logger.info("Compiling the model...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

        # --- Callbacks ---
        logger.info(f"Setting up callbacks: EarlyStopping (patience={patience}), EpochLogger")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min', restore_best_weights=True, verbose=1)
        # Pass progress info to the logger callback
        epoch_logger_callback = EpochLogger(epochs, major_step_num, total_major_steps)

        # --- Model Training ---
        logger.info(f"Starting model training (Max Epochs={epochs})...")
        history = model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping, epoch_logger_callback], # Use the custom logger
            batch_size=batch_size,
            verbose=0 # Set to 0 to rely on custom logger
        )
        actual_epochs = len(history.history['loss']); logger.info(f"Training finished after {actual_epochs} epochs.")
        if early_stopping.stopped_epoch > 0:
            # The best epoch is patience epochs before the stopped epoch
            best_epoch = max(1, early_stopping.stopped_epoch + 1 - patience)
            logger.info(f"Early stopping triggered. Restoring weights from epoch {best_epoch}.")
        else:
             logger.info("Training completed full epochs or stopped manually.")


        # --- Evaluate on Test Data ---
        if is_test_data_valid:
            logger.info("Evaluating final model on Test Data...")
            results = model.evaluate(X_test_scaled, y_test, verbose=0, batch_size=batch_size*4) # Use larger batch for eval
            if results and len(results) >= 3:
                # Use metric names from compile step if possible
                loss_name = model.metrics_names[0]
                mae_name = model.metrics_names[1]
                mse_name = model.metrics_names[2] # Assuming order loss, mae, mse
                logger.info(f"  Test {loss_name}: {results[0]:.6f}, Test {mae_name}: {results[1]:.6f}, Test {mse_name}(RMSE): {np.sqrt(results[2]):.6f}")
            else: logger.warning(f"Model evaluation on test set returned unexpected results: {results}")
        else: logger.info("Skipping test data evaluation.")

        # --- Save Model ---
        logger.info(f"Saving trained model to {model_save_path}...")
        model.save(model_save_path) # Save in Keras format

        # --- Plot Training History ---
        logger.info("Plotting training history...")
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6));
            axes[0].plot(history.history['loss'], label='Train Loss (MSE)'); axes[0].plot(history.history['val_loss'], label='Validation Loss (MSE)'); axes[0].set_title('Model Loss (MSE)'); axes[0].set_ylabel('Loss (MSE)'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True);
            axes[1].plot(history.history['mae'], label='Train MAE'); axes[1].plot(history.history['val_mae'], label='Validation MAE'); axes[1].set_title('Model MAE'); axes[1].set_ylabel('MAE'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True);
            plt.tight_layout(); plt.savefig(plot_save_path, dpi=150); plt.close(fig); logger.info(f"Training history plot saved to {plot_save_path}")
        except Exception as plot_err: logger.warning(f"Could not plot training history: {plot_err}", exc_info=False)

        logger.info("DNN training and evaluation complete.")
        return model_save_path
    except FileNotFoundError: logger.error(f"Processed data files not found in '{processed_data_dir}'."); raise
    except ValueError as ve: logger.error(f"DNN training data error: {ve}", exc_info=True); raise
    except ImportError: logger.critical("TensorFlow or required Keras components not installed."); raise # Specific error for TF
    except Exception as e: logger.error(f"Unexpected error during DNN training: {e}", exc_info=True); raise
# === END SECTION 4 ===
# Place this function definition earlier in the file, e.g., in SECTION 5 (Part 1)

def get_trades(latest_scores_dict, current_holdings_set, k, n):
    """Determines buy/sell trades based on scores, holdings, k, and n."""
    selected_buys = set(); selected_sells = set()
    if not latest_scores_dict: return selected_buys, selected_sells
    try:
        valid_scores = {str(stk): float(scr) for stk, scr in latest_scores_dict.items() if pd.notna(scr) and isinstance(scr, (int, float, np.number))}
        if not valid_scores: return selected_buys, selected_sells
        sorted_stocks_with_scores = sorted(valid_scores.items(), key=lambda item: item[1], reverse=True)
        stock_score_map = dict(sorted_stocks_with_scores); ranked_stocks_list = [s[0] for s in sorted_stocks_with_scores]
        target_portfolio = set(ranked_stocks_list[:k]); current_holdings_str = {str(s) for s in current_holdings_set}
        required_sells_candidates = current_holdings_str - target_portfolio
        required_sells_ranked = sorted([(s, stock_score_map.get(s, -np.inf)) for s in required_sells_candidates], key=lambda item: item[1])
        required_sells_set = {s[0] for s in required_sells_ranked}
        required_buys_candidates = target_portfolio - current_holdings_str
        required_buys_ranked = sorted([(s, stock_score_map.get(s, -np.inf)) for s in required_buys_candidates], key=lambda item: item[1], reverse=True)
        required_buys_set = {s[0] for s in required_buys_ranked}
        total_required_trades = len(required_sells_set) + len(required_buys_set)
        if total_required_trades <= n:
            selected_sells = required_sells_set; selected_buys = required_buys_set
        else:
            trades_done = 0
            for stock, _ in required_sells_ranked: # Prioritize sells
                if trades_done < n: selected_sells.add(stock); trades_done += 1;
                else: break
            if trades_done < n: # Add buys if limit not reached
                for stock, _ in required_buys_ranked:
                    if trades_done < n: selected_buys.add(stock); trades_done += 1;
                    else: break
    except Exception as e:
        logger.error(f"ERROR in get_trades logic: {e}", exc_info=True)
        return set(), set() # Return empty sets on error
    return selected_buys, selected_sells
# Place this function definition earlier in the file, e.g., in SECTION 5 (Part 1)

def get_trades(latest_scores_dict, current_holdings_set, k, n):
    """Determines buy/sell trades based on scores, holdings, k, and n."""
    selected_buys = set(); selected_sells = set()
    if not latest_scores_dict: return selected_buys, selected_sells
    try:
        valid_scores = {str(stk): float(scr) for stk, scr in latest_scores_dict.items() if pd.notna(scr) and isinstance(scr, (int, float, np.number))}
        if not valid_scores: return selected_buys, selected_sells
        sorted_stocks_with_scores = sorted(valid_scores.items(), key=lambda item: item[1], reverse=True)
        stock_score_map = dict(sorted_stocks_with_scores); ranked_stocks_list = [s[0] for s in sorted_stocks_with_scores]
        target_portfolio = set(ranked_stocks_list[:k]); current_holdings_str = {str(s) for s in current_holdings_set}
        required_sells_candidates = current_holdings_str - target_portfolio
        required_sells_ranked = sorted([(s, stock_score_map.get(s, -np.inf)) for s in required_sells_candidates], key=lambda item: item[1])
        required_sells_set = {s[0] for s in required_sells_ranked}
        required_buys_candidates = target_portfolio - current_holdings_str
        required_buys_ranked = sorted([(s, stock_score_map.get(s, -np.inf)) for s in required_buys_candidates], key=lambda item: item[1], reverse=True)
        required_buys_set = {s[0] for s in required_buys_ranked}
        total_required_trades = len(required_sells_set) + len(required_buys_set)
        if total_required_trades <= n:
            selected_sells = required_sells_set; selected_buys = required_buys_set
        else:
            trades_done = 0
            for stock, _ in required_sells_ranked: # Prioritize sells
                if trades_done < n: selected_sells.add(stock); trades_done += 1;
                else: break
            if trades_done < n: # Add buys if limit not reached
                for stock, _ in required_buys_ranked:
                    if trades_done < n: selected_buys.add(stock); trades_done += 1;
                    else: break
    except Exception as e:
        logger.error(f"ERROR in get_trades logic: {e}", exc_info=True)
        return set(), set() # Return empty sets on error
    return selected_buys, selected_sells
# === SECTION 5: Backtesting, Grid Search, and Analysis ===
# ... (Keep get_trades, calculate_metrics, calculate_benchmark_comparison_metrics as adjusted) ...
def run_simulation(results_df, k, n, period_name="", risk_free_rate=0.0, trading_days_year=252):
     """Runs the trading simulation based on predicted scores and actual returns."""
     logger.info(f"--- Running Simulation ({period_name}): k={k}, n={n} ---")
     start_time_sim = time.time(); current_holdings = set(); portfolio_daily_returns = {}
     if 'Date' in results_df.columns and not isinstance(results_df.index, pd.DatetimeIndex): results_df = results_df.set_index('Date')
     if not isinstance(results_df.index, pd.DatetimeIndex): raise ValueError("Simulation DF must have DatetimeIndex.")
     results_df = results_df.sort_index();
     if not results_df.index.is_monotonic_increasing: logger.warning(f"Sim input index not monotonic ({period_name}). Sorting."); results_df = results_df.sort_index()
     required_sim_cols = ['stock_id', 'predicted_score', 'actual_return']; missing_sim_cols = [col for col in required_sim_cols if col not in results_df.columns]
     if missing_sim_cols: raise ValueError(f"Simulation DF missing columns: {missing_sim_cols}")
     results_df['stock_id'] = results_df['stock_id'].astype(str); results_df['predicted_score'] = pd.to_numeric(results_df['predicted_score'], errors='coerce'); results_df['actual_return'] = pd.to_numeric(results_df['actual_return'], errors='coerce')
     unique_dates = results_df.index.unique().dropna().sort_values()
     if len(unique_dates) < 2: logger.warning(f"Not enough dates ({len(unique_dates)}) for sim ({period_name})."); empty_returns = pd.Series(dtype=np.float64, index=pd.to_datetime([])); return empty_returns, calculate_metrics(empty_returns) # Return empty series with datetime index

     # --- Grid Search / Simulation Loop ---
     task_context_sim = current_task._get_current_object() if hasattr(current_task, '_get_current_object') else None
     total_sim_days = len(unique_dates) - 1
     log_interval_sim = max(1, total_sim_days // 10) # Log roughly 10 times

     for i in range(total_sim_days):
         current_date = unique_dates[i]; next_date = unique_dates[i+1]; latest_scores_dict = {}
         try: # Get scores from current date
             daily_data_slice = results_df.loc[[current_date]];
             if not daily_data_slice.empty: latest_scores_dict = pd.Series(daily_data_slice.predicted_score.values, index=daily_data_slice.stock_id).dropna().to_dict()
         except Exception as score_err: logger.error(f"Sim Error Get Scores date={current_date}: {score_err}", exc_info=False)
         try: executed_buys, executed_sells = get_trades(latest_scores_dict, current_holdings, k, n) # Determine trades
         except Exception as trade_err: logger.error(f"Sim Error Get Trades date={current_date}: {trade_err}", exc_info=False); executed_buys, executed_sells = set(), set()
         holdings_after_sell = current_holdings - executed_sells; next_day_holdings = holdings_after_sell.union(executed_buys) # Determine next day's holdings
         daily_return = 0.0 # Calculate return for next day based on these holdings
         if next_day_holdings:
             try:
                 next_day_returns_slice = results_df.loc[[next_date]]
                 if not next_day_returns_slice.empty:
                     held_returns = next_day_returns_slice[next_day_returns_slice['stock_id'].isin(next_day_holdings)]['actual_return']
                     valid_returns = held_returns.dropna()
                     if not valid_returns.empty: daily_return = valid_returns.mean()
                     if len(valid_returns) < len(held_returns): logger.debug(f"NaN return for some held stocks on {next_date.date()}")
             except Exception as ret_err: logger.error(f"Sim Error Calc Return date={next_date}: {ret_err}", exc_info=False)
         portfolio_daily_returns[next_date] = daily_return; current_holdings = next_day_holdings.copy() # Update holdings for next loop

         # --- DEBUG: Comment out Celery progress for inner sim loop (can be too noisy) ---
         # if task_context_sim and (i + 1) % log_interval_sim == 0:
             # try:
                 # # Update progress based on simulation day completion
                 # sim_progress = int(100 * (i + 1) / total_sim_days)
                 # # Map to overall progress (Assume this is within Grid Search or Ext Sim step)
                 # # This requires knowing the major step number... needs refactoring if progress is needed here.
                 # overall_progress = ... # Calculation depends on context
                 # meta = {'step': f'Simulating ({period_name}) Day {i+1}/{total_sim_days}', 'progress': overall_progress}
                 # task_context_sim.update_state(state='PROGRESS', meta=meta)
             # except Exception as e:
                 # logger.warning(f"Could not update Celery task state during simulation: {e}")
         # --- END DEBUG ---
         pass

     daily_returns_strategy = pd.Series(portfolio_daily_returns).sort_index(); strategy_metrics = calculate_metrics(daily_returns_strategy, risk_free_rate, trading_days_year); sim_time = time.time() - start_time_sim
     logger.info(f"Finished Sim ({period_name}): k={k}, n={n}. CumRet={strategy_metrics.get('Cumulative Return', np.nan):.2%}, Sharpe={strategy_metrics.get('Sharpe Ratio', np.nan):.3f}. Time={sim_time:.1f}s")
     return daily_returns_strategy, strategy_metrics

# Place this function definition earlier in the file, e.g., after calculate_metrics

def calculate_benchmark_comparison_metrics(strategy_returns, benchmark_returns, risk_free_rate=0.0, trading_days_year=252):
    """Calculates Beta, Alpha, Correlation, Tracking Error vs benchmark."""
    comp_metrics={'Beta':None,'Alpha (Jensen)':None,'Correlation':None,'Tracking Error':None}; min_data_points=20
    if strategy_returns is None or benchmark_returns is None or strategy_returns.empty or benchmark_returns.empty: return comp_metrics
    s_ret=pd.Series(strategy_returns,name='strategy') if not isinstance(strategy_returns,pd.Series) else strategy_returns.rename('strategy'); b_ret=pd.Series(benchmark_returns,name='benchmark') if not isinstance(benchmark_returns,pd.Series) else benchmark_returns.rename('benchmark')
    s_ret=pd.to_numeric(s_ret,errors='coerce'); b_ret=pd.to_numeric(b_ret,errors='coerce'); aligned_df=pd.DataFrame({'strategy':s_ret,'benchmark':b_ret}).dropna()
    if len(aligned_df)<2: return comp_metrics # Need at least 2 points for corr/TE
    strat_ret_aligned=aligned_df['strategy']; bench_ret_aligned=aligned_df['benchmark']
    try: correlation=strat_ret_aligned.corr(bench_ret_aligned); comp_metrics['Correlation']=correlation if pd.notna(correlation) else None
    except Exception as e: logger.warning(f"Error calc Correlation: {e}")
    try: active_return=strat_ret_aligned-bench_ret_aligned; tracking_error_ann=active_return.std(ddof=1)*np.sqrt(trading_days_year); comp_metrics['Tracking Error']=tracking_error_ann if pd.notna(tracking_error_ann) else None
    except Exception as e: logger.warning(f"Error calc TrackErr: {e}")
    if len(aligned_df)<min_data_points: logger.warning(f"Comp Metrics: Not enough points ({len(aligned_df)}) for Beta/Alpha."); return comp_metrics # Return only Corr/TE if too few points
    try:
        daily_risk_free=(1+risk_free_rate)**(1/trading_days_year)-1; strat_excess=strat_ret_aligned-daily_risk_free; bench_excess=bench_ret_aligned-daily_risk_free; benchmark_variance=bench_excess.var(ddof=1)
        if pd.isna(benchmark_variance) or benchmark_variance<1e-12: logger.warning(f"Comp Metrics: Bench variance {benchmark_variance:.2e}. Cannot calc Beta/Alpha."); return comp_metrics
        bench_excess_with_const=sm.add_constant(bench_excess,prepend=False,has_constant='add'); model=sm.OLS(strat_excess,bench_excess_with_const).fit()
        beta=model.params.get('benchmark',np.nan); comp_metrics['Beta']=beta if pd.notna(beta) else None
        alpha_daily=model.params.get('const',np.nan); alpha_annualized=None
        if pd.notna(alpha_daily): alpha_annualized=alpha_daily*trading_days_year
        comp_metrics['Alpha (Jensen)']=alpha_annualized if pd.notna(alpha_annualized) else None
    except Exception as e: logger.error(f"Error during OLS regression: {e}",exc_info=True)
    # Final cleanup for JSON
    for key,value in comp_metrics.items():
        if isinstance(value,(np.number,np.bool_)): comp_metrics[key]=value.item()
        elif pd.isna(value): comp_metrics[key]=None
        elif isinstance(value,float) and not np.isfinite(value): comp_metrics[key]=None
    return comp_metrics

def backtest_and_analyze(
    processed_data_dir, model_path, scaler_path, feature_list_path,
    train_end_date_str, validation_end_date_str,
    k_candidates_list, n_candidates_list, # Use explicit list names
    validation_metric, n_to_test,
    benchmark_ticker, tracker_fund_ticker,
    risk_free, trading_days_year, forward_return_period,
    batch_size, output_dir, seed,
    major_step_num, total_major_steps # Pass progress info
    ):
    """Performs backtesting, grid search, and comparison analysis."""
    logger.info("--- Starting Backtesting, Grid Search, and Analysis ---")
    set_seeds(seed) # Set seed for this step
    try:
        # --- Step 1: Load Data & Model ---
        logger.info("Loading model, scaler, features, data arrays, and context...")
        try: # Wrap loading in try/except
            # Import TF locally if needed for load_model
            from tensorflow.keras.models import load_model
            model = load_model(model_path); scaler = joblib.load(scaler_path)
            with open(feature_list_path, 'r') as f: features = [line.strip() for line in f if line.strip()]
            X_val_scaled=np.load(os.path.join(processed_data_dir,'X_val_scaled.npy')); y_val=np.load(os.path.join(processed_data_dir,'y_val_continuous.npy'))
            X_test_scaled=np.load(os.path.join(processed_data_dir,'X_test_scaled.npy')); y_test=np.load(os.path.join(processed_data_dir,'y_test_continuous.npy'))
            val_context=pd.read_pickle(os.path.join(processed_data_dir,'val_context.pkl')); test_context=pd.read_pickle(os.path.join(processed_data_dir,'test_context.pkl'))
        except FileNotFoundError as fnf_err: logger.error(f"Failed loading file: {fnf_err}"); raise
        except ImportError: logger.critical("TensorFlow needed for load_model not installed."); raise
        logger.info(f"Loaded data shapes: X_val={X_val_scaled.shape}, y_val={y_val.shape}, X_test={X_test_scaled.shape}, y_test={y_test.shape}")
        logger.info(f"Using Features: {features}")

        # --- Step 2: Generate Predictions ---
        logger.info("Generating predictions...")
        pred_batch_size = batch_size * 4 # Use larger batch for prediction
        validation_predictions = model.predict(X_val_scaled, batch_size=pred_batch_size).flatten()
        test_predictions = model.predict(X_test_scaled, batch_size=pred_batch_size).flatten() if X_test_scaled.shape[0] > 0 else np.array([], dtype=float)
        if test_predictions.size == 0: logger.info("Test set empty, skipping test predictions.")

        # --- Step 3: Prepare Simulation DFs ---
        logger.info("Preparing simulation dataframes...")
        # ... (Code as provided previously - ensure correct context/prediction assignment) ...
        validation_results_df=val_context.copy(); validation_results_df['predicted_score']=validation_predictions; validation_results_df['actual_return']=y_val; validation_results_df['Date']=pd.to_datetime(validation_results_df['Date']).dt.normalize(); validation_results_df=validation_results_df.set_index('Date').sort_index()
        test_results_df=pd.DataFrame(columns=['Date','stock_id','predicted_score','actual_return']).set_index('Date')
        if test_predictions.size>0 and not test_context.empty: test_results_df=test_context.copy(); test_results_df['predicted_score']=test_predictions; test_results_df['actual_return']=y_test; test_results_df['Date']=pd.to_datetime(test_results_df['Date']).dt.normalize(); test_results_df=test_results_df.set_index('Date').sort_index()
        val_rows_before=len(validation_results_df); validation_results_df.dropna(subset=['actual_return'],inplace=True); logger.info(f"Val sim: Dropped {val_rows_before - len(validation_results_df)} NaN return rows.")
        if not test_results_df.empty: test_rows_before=len(test_results_df); test_results_df.dropna(subset=['actual_return'],inplace=True); logger.info(f"Test sim: Dropped {test_rows_before - len(test_results_df)} NaN return rows.")
        validation_unique_dates=validation_results_df.index.unique().dropna(); test_unique_dates=test_results_df.index.unique().dropna() if not test_results_df.empty else pd.to_datetime([])
        if validation_unique_dates.empty: raise ValueError("Val sim DF empty.")
        logger.info(f"Val sim ready: {validation_results_df.shape}, Dates: {validation_unique_dates.min().date()} to {validation_unique_dates.max().date()}")
        logger.info(f"Test sim ready: {test_results_df.shape}, Dates: {(test_unique_dates.min().date() if not test_unique_dates.empty else 'N/A')} to {(test_unique_dates.max().date() if not test_unique_dates.empty else 'N/A')}")


        # --- Step 4: Calculate Comparison Metrics ---
        logger.info(f"Calculating Benchmark ({benchmark_ticker}) & Tracker ({tracker_fund_ticker}) Metrics...")
        # ... (Refined comparison metric calculation block from previous response) ...
        val_benchmark_daily_returns=None; test_benchmark_daily_returns=None; val_tracker_daily_returns=None; test_tracker_daily_returns=None
        val_benchmark_metrics={}; test_benchmark_metrics={}; val_tracker_metrics={}; test_tracker_metrics={}
        val_benchmark_cumulative=None; test_benchmark_cumulative=None; val_tracker_cumulative=None; test_tracker_cumulative=None
        try:
             hsi_start_date=validation_unique_dates.min()-pd.Timedelta(days=10); max_date_needed=test_unique_dates.max() if not test_unique_dates.empty else validation_unique_dates.max(); hsi_end_date=max_date_needed+pd.Timedelta(days=2)
             tickers_to_fetch=list(set([benchmark_ticker,tracker_fund_ticker])); logger.info(f"Fetching {tickers_to_fetch} from {hsi_start_date.date()} to {hsi_end_date.date()}")
             comparison_data=yf.download(tickers_to_fetch,start=hsi_start_date,end=hsi_end_date,progress=False,ignore_tz=True,auto_adjust=True,timeout=30)
             if comparison_data.empty or comparison_data.shape[0]<2: logger.warning(f"Could not download sufficient comparison data."); val_benchmark_metrics=calculate_metrics(pd.Series(dtype=float)); test_benchmark_metrics=calculate_metrics(pd.Series(dtype=float)); val_tracker_metrics=calculate_metrics(pd.Series(dtype=float)); test_tracker_metrics=calculate_metrics(pd.Series(dtype=float))
             else:
                  prices_df = pd.DataFrame(); # Logic to extract prices_df from comparison_data
                  if isinstance(comparison_data.columns, pd.MultiIndex) and 'Close' in comparison_data.columns.levels[0]: prices_df = comparison_data['Close'].copy()
                  elif isinstance(comparison_data.columns, pd.Index):
                      present_tickers = [t for t in tickers_to_fetch if t in comparison_data.columns]
                      if len(present_tickers) == len(tickers_to_fetch): prices_df = comparison_data[tickers_to_fetch].copy()
                      elif 'Close' in comparison_data.columns:
                          if len(tickers_to_fetch)==1: prices_df=comparison_data[['Close']].rename(columns={'Close':tickers_to_fetch[0]})
                          else: prices_df=comparison_data[['Close']]; logger.warning("Flat comparison data, using 'Close'.")
                      else: logger.warning("Cannot find Close/ticker columns in flat comparison data.")
                  # Process prices_df if not empty
                  if not prices_df.empty:
                       prices_df.index = pd.to_datetime(prices_df.index).normalize(); prices_df = prices_df.dropna(how='any')
                       if len(prices_df) >= 2:
                            daily_returns_raw = prices_df.pct_change().dropna()
                            def calculate_period_metrics(ticker, period_dates, period_name): # Inner helper improved
                                 if ticker not in daily_returns_raw.columns: logger.warning(f"{ticker} not in comparison returns."); return None, calculate_metrics(pd.Series(dtype=float)), None
                                 aligned_returns = daily_returns_raw[ticker].reindex(period_dates).dropna()
                                 if aligned_returns.empty: logger.warning(f"No {ticker} data for {period_name} period."); return None, calculate_metrics(pd.Series(dtype=float)), None
                                 metrics = calculate_metrics(aligned_returns, risk_free, trading_days_year); logger.info(f"  {ticker} Perf ({period_name}): { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in metrics.items()} }"); cumulative = (1 + aligned_returns.fillna(0)).cumprod(); return aligned_returns, metrics, cumulative
                            val_benchmark_daily_returns, val_benchmark_metrics, val_benchmark_cumulative = calculate_period_metrics(benchmark_ticker, validation_unique_dates, "Validation")
                            val_tracker_daily_returns, val_tracker_metrics, val_tracker_cumulative = calculate_period_metrics(tracker_fund_ticker, validation_unique_dates, "Validation")
                            if not test_unique_dates.empty:
                                 test_benchmark_daily_returns, test_benchmark_metrics, test_benchmark_cumulative = calculate_period_metrics(benchmark_ticker, test_unique_dates, "Test")
                                 test_tracker_daily_returns, test_tracker_metrics, test_tracker_cumulative = calculate_period_metrics(tracker_fund_ticker, test_unique_dates, "Test")
                            else:
                                 logger.info("Skipping Test comparison calc as test dates are empty.")
                                 test_benchmark_metrics = calculate_metrics(pd.Series(dtype=float)); test_tracker_metrics = calculate_metrics(pd.Series(dtype=float))
                       else: logger.warning("Not enough comparison points after dropna."); # Set empty metrics
                  else: logger.warning("Could not extract prices from comparison data."); # Set empty metrics
        except Exception as e: logger.error(f"ERROR calculating comparison returns: {e}", exc_info=True); # Set empty metrics
        # Ensure metrics dicts exist even if calc failed
        if not val_benchmark_metrics: val_benchmark_metrics=calculate_metrics(pd.Series(dtype=float))
        if not test_benchmark_metrics: test_benchmark_metrics=calculate_metrics(pd.Series(dtype=float))
        if not val_tracker_metrics: val_tracker_metrics=calculate_metrics(pd.Series(dtype=float))
        if not test_tracker_metrics: test_tracker_metrics=calculate_metrics(pd.Series(dtype=float))


        # --- Step 5: Grid Search ---
        num_sims = len(k_candidates_list) * len(n_candidates_list)
        logger.info(f"--- Starting Grid Search ({num_sims} simulations) on VALIDATION SET ---")
        validation_run_results = []; grid_start_time = time.time(); sim_count = 0;
        task_context_grid = current_task._get_current_object() if hasattr(current_task, '_get_current_object') else None

        for k in k_candidates_list:
             if k <= 0: continue
             for n in n_candidates_list:
                  sim_count += 1;
                  if n <= 0: continue
                  # Run simulation for this k, n pair
                  _, strategy_metrics_val = run_simulation(validation_results_df.copy(), k, n, f"Validation(k={k},n={n})", risk_free, trading_days_year)
                  current_run_result = {'k': k, 'n': n}; current_run_result.update(strategy_metrics_val); validation_run_results.append(current_run_result)

                  # --- Celery Progress Update for Grid Search ---
                  if task_context_grid and num_sims > 0:
                       try:
                           # Calculate progress within this major step (Backtesting)
                           step_progress = int(100 * sim_count / num_sims) if num_sims > 0 else 100
                           step_start_progress = int(100 * (major_step_num - 1) / total_major_steps)
                           step_end_progress = int(100 * major_step_num / total_major_steps)
                           overall_progress = step_start_progress + int(step_progress * (step_end_progress - step_start_progress) / 100)
                           meta = {'step': f'Grid Search ({sim_count}/{num_sims}) k={k},n={n}', 'progress': min(overall_progress, 100)}
                           task_context_grid.update_state(state='PROGRESS', meta=meta)
                       except Exception as e:
                           logger.warning(f"Could not update Celery task state during grid search: {e}")
                  # --- End Celery Update ---

        grid_time = time.time() - grid_start_time; logger.info(f"--- Grid Search Complete ({grid_time:.2f} seconds) ---")

        # --- Step 6: Select Top N Candidates ---
        logger.info(f"--- Selecting Top {n_to_test} based on Val '{validation_metric}' ---"); best_k_val = None; best_n_val = None; top_n_validation_candidates = []
        if not validation_run_results: raise ValueError("Validation grid search yielded no results.")
        results_val_metrics_df = pd.DataFrame(validation_run_results)
        if validation_metric not in results_val_metrics_df.columns: raise ValueError(f"Validation metric '{validation_metric}' not found.")
        results_val_metrics_df[validation_metric] = pd.to_numeric(results_val_metrics_df[validation_metric], errors='coerce')
        results_val_metrics_df = results_val_metrics_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[validation_metric])
        if results_val_metrics_df.empty: raise ValueError(f"No valid candidates after validation grid search (metric: {validation_metric}).")
        results_val_metrics_df.sort_values(by=validation_metric, ascending=False, inplace=True)
        num_to_select = min(n_to_test, len(results_val_metrics_df)); logger.info(f"Selecting Top {num_to_select} candidates...")
        for i in range(num_to_select):
            row = results_val_metrics_df.iloc[i]; k_cand = int(row['k']); n_cand = int(row['n']); metrics_cand = row.drop(['k', 'n']).to_dict(); top_n_validation_candidates.append({'k': k_cand, 'n': n_cand, 'val_metrics': metrics_cand})
            if i == 0: best_k_val = k_cand; best_n_val = n_cand
        if best_k_val is None: raise ValueError("Could not determine best K/N.")
        logger.info(f"Selected. Best Validation: k={best_k_val}, n={best_n_val}")
        cols_to_log = ['k', 'n', validation_metric, 'Cumulative Return', 'Sharpe Ratio', 'Max Drawdown']; existing_cols_log = [c for c in cols_to_log if c in results_val_metrics_df.columns]; logger.info(f"\n--- Top {num_to_select} Validation Candidates Performance ---\n" + results_val_metrics_df[existing_cols_log].head(num_to_select).to_string(index=False, float_format="%.4f"))

        # --- Step 7: Evaluate on Test Set ---
        logger.info(f"--- Evaluating Top {len(top_n_validation_candidates)} Candidates on TEST SET ---"); test_set_results = []
        if not test_results_df.empty and top_n_validation_candidates:
             for candidate in top_n_validation_candidates:
                 k_test=candidate['k']; n_test=candidate['n']; logger.info(f"--- Running Test Sim: k={k_test}, n={n_test} ---")
                 test_daily_returns, test_metrics = run_simulation(test_results_df.copy(), k_test, n_test, f"Test(k={k_test},n={n_test})", risk_free, trading_days_year)
                 # Calculate comparison vs *aligned* benchmark returns for the test period
                 test_comparison_metrics = calculate_benchmark_comparison_metrics(test_daily_returns, test_benchmark_daily_returns, risk_free, trading_days_year)
                 full_test_metrics = {**test_metrics, **test_comparison_metrics}
                 test_set_results.append({'k': k_test, 'n': n_test, 'test_metrics': full_test_metrics, 'test_daily_returns': test_daily_returns, 'validation_metrics': candidate['val_metrics']})
        elif test_results_df.empty: logger.warning("Skipping test set eval - test data empty.")
        else: logger.warning("Skipping test set eval - no candidates.")

        # --- Step 8: Log Performance Summary ---
        logger.info("\n\n--- === BACKTEST PERFORMANCE SUMMARY (Log) === ---")
        logger.info(f"\n--- Test Set Benchmark ({benchmark_ticker}) Perf ---")
        if test_benchmark_metrics and any(v is not None for v in test_benchmark_metrics.values()): logger.info(f"{ {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in test_benchmark_metrics.items()} }")
        else: logger.info("  (N/A or No Data)")
        logger.info(f"\n--- Test Set Tracker ({tracker_fund_ticker}) Perf ---")
        if test_tracker_metrics and any(v is not None for v in test_tracker_metrics.values()): logger.info(f"{ {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in test_tracker_metrics.items()} }")
        else: logger.info("  (N/A or No Data)")
        logger.info(f"\n--- Test Set Perf Comparison for Top {len(test_set_results)} Candidate(s) ---")
        if test_set_results: # Display test results table...
             report_data_log = [];
             for i, result in enumerate(test_set_results):
                  test_m=result.get('test_metrics',{}); val_m=result.get('validation_metrics',{})
                  # Ensure keys exist before accessing
                  row = {
                      'Rank(Val)': i + 1, 'k': result.get('k', 'N/A'), 'n': result.get('n', 'N/A'),
                      'Test Sharpe': test_m.get('Sharpe Ratio'), 'Test CumRet': test_m.get('Cumulative Return'),
                      'Test MaxDD': test_m.get('Max Drawdown'), 'Test Beta': test_m.get('Beta'),
                      'Test Alpha': test_m.get('Alpha (Jensen)'), 'Test TE': test_m.get('Tracking Error'),
                      f'Val {validation_metric}': val_m.get(validation_metric)
                  }
                  report_data_log.append(row)
             report_df_log = pd.DataFrame(report_data_log); logger.info("\nTest Performance Table:\n" + report_df_log.to_string(index=False, float_format="%.4f"))
             # Recommendation notes...
             logger.info("\n--- Recommendation Notes ---"); logger.info(f"Candidate k={test_set_results[0]['k']}, n={test_set_results[0]['n']} ranked #1 on Val ({validation_metric}).")
             top_cand_test_metrics = test_set_results[0].get('test_metrics', {}); top_cand_sharpe = top_cand_test_metrics.get('Sharpe Ratio'); bench_test_sharpe = test_benchmark_metrics.get('Sharpe Ratio')
             if isinstance(top_cand_sharpe, (int, float)) and isinstance(bench_test_sharpe, (int, float)) and np.isfinite(top_cand_sharpe) and np.isfinite(bench_test_sharpe):
                  if top_cand_sharpe > bench_test_sharpe: logger.info(f"  -> Higher Test Sharpe ({top_cand_sharpe:.4f}) vs Benchmark ({bench_test_sharpe:.4f}).")
                  else: logger.info(f"  -> Lower Test Sharpe ({top_cand_sharpe:.4f}) vs Benchmark ({bench_test_sharpe:.4f}).")
             else: logger.info(f"  -> Cannot compare Test Sharpe vs Benchmark. Strategy: {top_cand_sharpe}, Benchmark: {bench_test_sharpe}")
             logger.info("  -> Review full Test Table.")
        else: logger.info("  No test set results.")

        # --- Step 9: Plot Test Set Results ---
        logger.info(f"--- Plotting #1 Ranked Candidate on TEST Period ---")
        test_plot_filename_final = None
        if test_set_results: # Plotting code...
            result_to_plot = test_set_results[0]; k_plot=result_to_plot['k']; n_plot=result_to_plot['n']; daily_returns_plot = result_to_plot.get('test_daily_returns')
            if daily_returns_plot is not None and not daily_returns_plot.empty:
                try: # Plot generation logic... save fig, close fig
                     plt.style.use('seaborn-v0_8-darkgrid'); fig, ax = plt.subplots(figsize=(14, 8)); strategy_cumulative_final = (1 + daily_returns_plot.fillna(0)).cumprod(); ax.plot(strategy_cumulative_final.index, strategy_cumulative_final, label=f'#1 Val Cand (k={k_plot},n={n_plot}) Test', lw=2.5, color='darkgreen', zorder=10);
                     # Plot benchmark using TEST period cumulative data
                     if test_benchmark_cumulative is not None and not test_benchmark_cumulative.empty: common_idx=strategy_cumulative_final.index.intersection(test_benchmark_cumulative.index); ax.plot(test_benchmark_cumulative.loc[common_idx].index, test_benchmark_cumulative.loc[common_idx], label=f'{benchmark_ticker} Benchmark', linestyle='--', lw=2, color='darkblue', zorder=5)
                     else: logger.warning("Cannot plot Test Benchmark - data missing.")
                     # Plot tracker using TEST period cumulative data
                     if test_tracker_cumulative is not None and not test_tracker_cumulative.empty: common_idx=strategy_cumulative_final.index.intersection(test_tracker_cumulative.index); ax.plot(test_tracker_cumulative.loc[common_idx].index, test_tracker_cumulative.loc[common_idx], label=f'{tracker_fund_ticker} Tracker', linestyle=':', lw=2, color='firebrick', zorder=4)
                     else: logger.warning("Cannot plot Test Tracker - data missing.")
                     # Formatting...
                     start_str=strategy_cumulative_final.index.min().strftime('%Y-%m-%d'); end_str=strategy_cumulative_final.index.max().strftime('%Y-%m-%d'); plot_title = f'#1 Val Cand ({validation_metric}) - TEST SET vs Comparisons\n{start_str} to {end_str}'; ax.set_title(plot_title, fontsize=14); ax.set_ylabel('Cumulative Return (Log Scale)', fontsize=13); ax.set_xlabel('Date', fontsize=13); ax.set_yscale('log'); ax.legend(loc='best', fontsize=11); ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1, symbol='%')); ax.grid(True, which='both', linestyle='--', linewidth=0.5); plt.xticks(rotation=30, ha='right'); plt.tight_layout();
                     test_plot_filename = f'dnn_backtest_TEST_SET_Top1ValMetric_k{k_plot}_n{n_plot}_cumulative_return_log.png'; plot_full_path = os.path.join(output_dir, test_plot_filename); plt.savefig(plot_full_path, dpi=150); plt.close(fig); logger.info(f"Test set plot saved: {plot_full_path}"); test_plot_filename_final = test_plot_filename
                except Exception as e: logger.error(f"Error during Test Set plotting: {e}", exc_info=True)
            else: logger.warning(f"Skipping Test Set plot: No returns data for #1 candidate.")
        else: logger.warning("Skipping Test Set plot: No test results.")

        # --- Step 10: Return Results ---
        return {
            "best_k": best_k_val, "best_n": best_n_val,
            "test_results_top_candidates": test_set_results, # List of dicts with metrics/returns
            "test_benchmark_daily_returns": test_benchmark_daily_returns, # Aligned returns series (or None)
            "test_benchmark_metrics": test_benchmark_metrics, # Dict of base metrics
            "test_tracker_daily_returns": test_tracker_daily_returns, # Aligned returns series (or None)
            "test_tracker_metrics": test_tracker_metrics, # Dict of base metrics
            "test_period_plot_relpath": test_plot_filename_final # Relative path string (or None)
        }

    except FileNotFoundError as fnf: logger.error(f"File not found during backtesting: {fnf}", exc_info=True); raise
    except ValueError as ve: logger.error(f"Data/parameter error during backtesting: {ve}", exc_info=True); raise
    except ImportError: logger.critical("TensorFlow/Keras needed for load_model not installed."); raise
    except Exception as e: logger.error(f"Unexpected error during backtesting: {e}", exc_info=True); raise
# === END SECTION 5 ===


# === SECTION 6: Extended Simulation and Reporting Data Preparation ===
# --- get_holdings_on_date function ---
def get_holdings_on_date(results_df, k, n, target_date_dt):
    """Determines the set of holdings at the START of a given target date."""
    logger.debug(f"Simulating holdings for start of {target_date_dt.date()} (k={k}, n={n})...")
    current_holdings = set()
    if results_df is None or results_df.empty: logger.warning("get_holdings: Input DF is empty."); return current_holdings
    if not isinstance(results_df.index, pd.DatetimeIndex): logger.error("get_holdings: results_df needs DatetimeIndex."); return current_holdings
    if not results_df.index.is_monotonic_increasing: logger.warning("get_holdings: index not monotonic. Sorting."); results_df = results_df.sort_index()
    required_cols = ['stock_id', 'predicted_score'];
    if not all(col in results_df.columns for col in required_cols): logger.error(f"get_holdings: missing cols: {required_cols}"); return current_holdings
    # Ensure correct types, handle potential NaNs introduced if columns were missing
    results_df['stock_id'] = results_df['stock_id'].astype(str) if 'stock_id' in results_df else None
    results_df['predicted_score'] = pd.to_numeric(results_df['predicted_score'], errors='coerce') if 'predicted_score' in results_df else np.nan
    results_df = results_df.dropna(subset=['stock_id']) # Cannot proceed without stock_id

    unique_dates_in_data = results_df.index.unique().dropna().sort_values()

    # Ensure timezone consistency for comparison
    is_target_aware = target_date_dt.tzinfo is not None and target_date_dt.tzinfo.utcoffset(target_date_dt) is not None
    is_index_aware = unique_dates_in_data.tz is not None

    target_date_comp = target_date_dt
    index_comp = unique_dates_in_data

    if is_target_aware and not is_index_aware:
        # Localize index to target's timezone or UTC for comparison
        try: index_comp = unique_dates_in_data.tz_localize(target_date_dt.tzinfo or 'UTC')
        except Exception as tz_err: logger.warning(f"Could not localize index for tz comparison: {tz_err}")
    elif not is_target_aware and is_index_aware:
        # Make target aware (e.g., UTC) or make index naive
        target_date_comp = target_date_dt.tz_localize('UTC') # Example: Use UTC
        # Or make index naive: index_comp = unique_dates_in_data.tz_localize(None)
    elif is_target_aware and is_index_aware and target_date_dt.tzinfo != unique_dates_in_data.tz:
         # Convert one timezone to match the other (e.g., target to index's tz)
         try: target_date_comp = target_date_dt.tz_convert(unique_dates_in_data.tz)
         except Exception as tz_err: logger.warning(f"Could not convert target date tz for comparison: {tz_err}")

    # Filter relevant dates using the potentially adjusted index/target
    relevant_dates = index_comp[index_comp < target_date_comp]

    if relevant_dates.empty: logger.debug(f"Holdings Sim: No days before {target_date_dt.date()}."); return current_holdings

    # --- Simulation loop ---
    for sim_date_comp in relevant_dates:
        # Find the original timestamp corresponding to the potentially timezone-adjusted one
        sim_date_orig = unique_dates_in_data[unique_dates_in_data == sim_date_comp]
        if sim_date_orig.empty: continue # Should not happen if filtering correctly

        latest_scores_dict = {}
        try: # Get scores from sim_date (using original index)
            daily_data_slice = results_df.loc[[sim_date_orig[0]]]; # Use the original timestamp
            if not daily_data_slice.empty: latest_scores_dict = pd.Series(daily_data_slice.predicted_score.values, index=daily_data_slice.stock_id).dropna().to_dict()
        except Exception as score_err: logger.warning(f"Holdings Sim WARNING: Get Scores {sim_date_orig[0].date()}: {score_err}")
        try: # Get trades based on scores and update holdings
            executed_buys, executed_sells = get_trades(latest_scores_dict, current_holdings, k, n); holdings_after_sell = current_holdings - executed_sells; current_holdings = holdings_after_sell.union(executed_buys)
        except Exception as trade_err: logger.warning(f"Holdings Sim WARNING: Get Trades {sim_date_orig[0].date()}: {trade_err}")
    logger.debug(f"Holdings calculated for START of {target_date_dt.date()}: {len(current_holdings)} stocks.")
    return current_holdings
def calculate_metrics(daily_returns, risk_free_rate=0.0, trading_days_year=252):
    """Calculates performance metrics for a daily returns series."""
    metrics={'Cumulative Return':None,'Annualized Return':None,'Annualized Volatility':None,'Sharpe Ratio':None,'Sortino Ratio':None,'Max Drawdown':None,'Trading Days':0,'Positive Days %':None,'risk_free_rate':risk_free_rate}
    if daily_returns is None or not isinstance(daily_returns,pd.Series) or daily_returns.empty: return metrics
    daily_returns=pd.to_numeric(daily_returns,errors='coerce').dropna()
    if daily_returns.empty: return metrics
    n_days=len(daily_returns); metrics['Trading Days']=n_days
    if n_days<2: # Handle short series
        try:
             metrics['Cumulative Return']=(1+daily_returns).prod()-1 if n_days > 0 else 0.0
             metrics['Positive Days %']=(daily_returns>1e-9).mean()*100 if n_days>0 else 0.0
        except Exception as e: logger.warning(f"Metric calc error (short series): {e}")
        for key,value in metrics.items(): # Ensure JSON safe
            if isinstance(value,(np.number,np.bool_)): metrics[key]=value.item()
            elif pd.isna(value): metrics[key]=None
        return metrics
    # Calculate metrics with error handling
    try: cumulative_return=(1+daily_returns).prod()-1; metrics['Cumulative Return']=cumulative_return if pd.notna(cumulative_return) else None
    except Exception as e: logger.warning(f"Metric calc error (CumRet): {e}")
    try:
        n_years=n_days/trading_days_year
        if n_years>0 and metrics['Cumulative Return'] is not None:
             if metrics['Cumulative Return']>-1: metrics['Annualized Return']=(1+metrics['Cumulative Return'])**(1/n_years)-1
             elif metrics['Cumulative Return']==-1: metrics['Annualized Return']=-1.0
    except Exception as e: logger.warning(f"Metric calc error (AnnRet): {e}")
    try: annualized_volatility=daily_returns.std(ddof=1)*np.sqrt(trading_days_year); metrics['Annualized Volatility']=annualized_volatility if pd.notna(annualized_volatility) else None
    except Exception as e: logger.warning(f"Metric calc error (AnnVol): {e}")
    ann_ret=metrics.get('Annualized Return'); ann_vol=metrics.get('Annualized Volatility')
    if ann_ret is not None and ann_vol is not None: # Sharpe Ratio
        try:
            if ann_vol>1e-9: metrics['Sharpe Ratio']=(ann_ret-risk_free_rate)/ann_vol
            elif abs(ann_ret-risk_free_rate)<1e-9: metrics['Sharpe Ratio']=0.0
            elif ann_ret>risk_free_rate: metrics['Sharpe Ratio']=np.inf
            else: metrics['Sharpe Ratio']=-np.inf
        except Exception as e: logger.warning(f"Metric calc error (Sharpe): {e}")
    if ann_ret is not None: # Sortino Ratio
        try:
            daily_target_return=(1+risk_free_rate)**(1/trading_days_year)-1; downside_returns=daily_returns[daily_returns<daily_target_return]; downside_deviation_ann=None
            if not downside_returns.empty: downside_diff_sq=np.square(downside_returns-daily_target_return); downside_std_dev_daily=np.sqrt(downside_diff_sq.mean()); downside_deviation_ann=downside_std_dev_daily*np.sqrt(trading_days_year);
            if downside_deviation_ann is not None and np.isclose(downside_deviation_ann,0): downside_deviation_ann=0.0
            if downside_deviation_ann is not None:
                 if downside_deviation_ann>1e-9: metrics['Sortino Ratio']=(ann_ret-risk_free_rate)/downside_deviation_ann
                 elif abs(ann_ret-risk_free_rate)<1e-9: metrics['Sortino Ratio']=0.0
                 elif ann_ret>risk_free_rate: metrics['Sortino Ratio']=np.inf
                 else: metrics['Sortino Ratio']=-np.inf
        except Exception as e: logger.warning(f"Metric calc error (Sortino): {e}")
    try: # Max Drawdown
        cumulative_wealth=(1+daily_returns).cumprod(); peak=cumulative_wealth.cummax(); drawdown=(cumulative_wealth-peak)/peak.replace(0,np.nan); max_drawdown=drawdown.min(); metrics['Max Drawdown']=max_drawdown if pd.notna(max_drawdown) else None
    except Exception as e: logger.warning(f"Metric calc error (MaxDD): {e}")
    try: # Positive Days %
        metrics['Positive Days %']=(daily_returns>1e-9).mean()*100 if n_days>0 else 0.0 # Correct calculation (returns 0-100)
    except Exception as e: logger.warning(f"Metric calc error (PosDays): {e}")
    # Final cleanup for JSON compatibility
    for key,value in metrics.items():
        if isinstance(value,(np.number,np.bool_)): metrics[key]=value.item()
        elif pd.isna(value): metrics[key]=None
        elif isinstance(value,float) and not np.isfinite(value): metrics[key]=None # Handle Inf/-Inf
    return metrics

# --- run_extended_simulation_and_prep_report_data ---
# (Includes the fix for the 'config' NameError by adding run_config parameter)
def run_extended_simulation_and_prep_report_data(
    original_filtered_data_csv, feature_names, tickers,
    model_path, scaler_path, best_k, best_n, forward_return_period,
    extend_end_date_str, total_capital, benchmark_ticker, tracker_fund_ticker,
    risk_free, trading_days_year, lookback_date_str,
    batch_size, output_dir, seed,
    major_step_num, total_major_steps,
    run_config: dict # Added run_config parameter
    ):
    logger.info("--- Starting Extended Simulation and Report Data Preparation ---")
    # Initialize variables
    extended_results_df = pd.DataFrame() # This holds combined data for simulation
    extended_daily_returns = None; extended_metrics_base = {}; dollar_equity_curve = None
    extended_benchmark_daily_returns_aligned = pd.Series(dtype=float); extended_tracker_daily_returns_aligned = pd.Series(dtype=float)
    extended_benchmark_metrics = {}; extended_tracker_metrics = {}; comparison_metrics = {}
    lookback_info = {}; next_day_info = {}; plot_paths = {}
    final_sim_start_date = None; final_sim_end_date = None
    task_context_ext = current_task._get_current_object() if hasattr(current_task, '_get_current_object') else None
    # Define temp path using output_dir
    TEMP_EXTENDED_RESULTS_PATH = os.path.join(output_dir, "temp_extended_results_for_snapshot.pkl")
    try:
        # --- Step 1: Load Original Data & Model/Scaler ---
        logger.info("Loading model, scaler, and original data...")
        # Import TF locally
        from tensorflow.keras.models import load_model
        try:
             model = load_model(model_path); scaler = joblib.load(scaler_path)
             original_data_df = pd.read_csv(original_filtered_data_csv, parse_dates=['Date'])
             original_data_df.sort_values(by=['stock_id', 'Date'], inplace=True)
             original_end_date = original_data_df['Date'].max(); logger.info(f"Original data ends: {original_end_date.date()}")
        except FileNotFoundError as fnf_load: logger.error(f"Loading failed: {fnf_load}"); raise
        except ImportError: logger.critical("TensorFlow needed for load_model not installed."); raise

        # --- Step 2: Fetch New Data ---
        logger.info("Fetching new data for extension period...")
        new_data_df = pd.DataFrame()
        try: extend_end_date = pd.to_datetime(extend_end_date_str)
        except ValueError as date_err: raise ValueError(f"Invalid EXTEND_SIMULATION_END_DATE_STR: {date_err}")
        # Fetch buffer for alpha calculation + new period
        alpha_lookback_days = 60; # Max lookback needed by alphas
        fetch_start_date_buffer = original_end_date - timedelta(days=alpha_lookback_days);
        fetch_start_date_actual = original_end_date + timedelta(days=1); # Start fetching day after original data ends
        fetch_end_date_yf = extend_end_date + timedelta(days=1) # yfinance excludes end

        if fetch_start_date_actual <= extend_end_date:
            logger.info(f"Fetching new data from {fetch_start_date_buffer.date()} to {fetch_end_date_yf.date()} (for alpha calc + sim period)...")
            new_data_list = []; fetch_errors_new = 0
            for ticker in tickers:
                try:
                    data_new = yf.download(ticker, start=fetch_start_date_buffer, end=fetch_end_date_yf, auto_adjust=True, progress=False, ignore_tz=True, timeout=20) # Adjust timeout if needed
                    if not data_new.empty:
                        if isinstance(data_new.columns, pd.MultiIndex): data_new.columns = data_new.columns.get_level_values(0)
                        data_new['stock_id'] = ticker; data_new.reset_index(inplace=True)
                        if 'Datetime' in data_new.columns: data_new.rename(columns={'Datetime': 'Date'}, inplace=True)
                        elif 'index' in data_new.columns and 'Date' not in data_new.columns: data_new.rename(columns={'index':'Date'}, inplace=True)
                        if 'Date' in data_new.columns:
                             data_new['Date'] = pd.to_datetime(data_new['Date']).dt.normalize()
                             # Keep necessary columns
                             cols_raw = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'stock_id']
                             new_data_list.append(data_new[[c for c in cols_raw if c in data_new.columns]])
                        else: logger.warning(f"Missing 'Date' for new data {ticker}"); fetch_errors_new += 1
                except Exception as e: logger.warning(f"WARN fetching {ticker}: {e}"); fetch_errors_new += 1
            if new_data_list:
                 new_data_df = pd.concat(new_data_list, ignore_index=True).sort_values(by=['stock_id', 'Date'])
                 logger.info(f"New raw data fetched. Shape: {new_data_df.shape}. Fetch errors: {fetch_errors_new}")
            else: logger.warning("No new data successfully fetched for the extension period.")
        else: logger.info("Extend date not after original data. No new data fetched.")

        # --- Step 3: Combine Data & Recalculate Alphas ---
        logger.info("Combining original and new data for full alpha calculation...")
        # Select only necessary columns from original data for alpha calc
        original_cols_for_alpha = ['Date', 'stock_id', 'Open', 'High', 'Low', 'Close', 'Volume']
        original_data_alpha_subset = original_data_df[[c for c in original_cols_for_alpha if c in original_data_df.columns]].copy()

        # Combine raw data (original subset + new fetched data)
        combined_raw_df = pd.concat([original_data_alpha_subset, new_data_df], ignore_index=True)
        # Critical: Sort and remove duplicates based on Date AND stock_id, keeping the LATEST entry (prefer new data if overlap)
        combined_raw_df.sort_values(by=['stock_id', 'Date'], inplace=True)
        combined_raw_df.drop_duplicates(subset=['Date', 'stock_id'], keep='last', inplace=True)
        logger.info(f"Combined raw shape for alpha calc: {combined_raw_df.shape}")

        # Calculate alphas on the *entire* combined history for each stock
        all_alpha_data_list = []; alpha_calc_errors = 0; grouped_combined_raw = combined_raw_df.groupby('stock_id'); num_groups = len(grouped_combined_raw); processed_groups = 0
        for ticker, ticker_data_full in grouped_combined_raw:
            processed_groups += 1;
            if processed_groups % 20 == 0 or processed_groups == num_groups: logger.info(f"Calculating alphas... {processed_groups}/{num_groups} ({ticker})")
            try:
                 _check_columns(ticker_data_full) # Check if OHLCV is present
                 alpha_data_all = calculate_all_alpha_factors(ticker_data_full.copy(), feature_names)
                 # Add price column (usually Close) after alpha calculation
                 alpha_data_all['price'] = alpha_data_all['Close'] if 'Close' in alpha_data_all else np.nan
                 # Select only needed columns AFTER calculation
                 cols_to_keep_alpha = ['Date', 'stock_id', 'price'] + feature_names
                 existing_cols_alpha = [col for col in cols_to_keep_alpha if col in alpha_data_all.columns]
                 all_alpha_data_list.append(alpha_data_all[existing_cols_alpha])
            except ValueError as ve: logger.warning(f"WARN skipping alpha {ticker}: {ve}"); alpha_calc_errors += 1
            except Exception as e: logger.warning(f"WARN alpha calc {ticker}: {e}", exc_info=False); alpha_calc_errors += 1

        if not all_alpha_data_list: raise RuntimeError("Alpha factor calculation failed for all tickers.")
        full_alpha_df = pd.concat(all_alpha_data_list, ignore_index=True).sort_values(by=['stock_id', 'Date'])
        logger.info(f"Full alpha dataset calculated. Shape: {full_alpha_df.shape}. Alpha calc errors: {alpha_calc_errors}")

        # Filter the full alpha dataset to the simulation period defined by run_config
        try:
            sim_start_date = pd.to_datetime(run_config['START_DATE_STR'])
            sim_end_date = pd.to_datetime(run_config['EXTEND_SIMULATION_END_DATE_STR'])
        except ValueError as date_err: raise ValueError(f"Invalid sim period date format: {date_err}")

        combined_data_df = full_alpha_df[(full_alpha_df['Date'] >= sim_start_date) & (full_alpha_df['Date'] <= sim_end_date)].copy()
        logger.info(f"Filtered alpha data for sim period ({sim_start_date.date()} to {sim_end_date.date()}). Shape: {combined_data_df.shape}")
        if combined_data_df.empty: raise ValueError(f"No data available for the simulation period ({sim_start_date.date()} to {sim_end_date.date()}) after alpha calculation.")

        # --- Step 4: Recalculate Returns & Predict ---
        logger.info("Recalculating forward returns & generating predictions on combined data...")
        combined_data_df = combined_data_df.sort_values(by=['stock_id', 'Date']) # Ensure order
        combined_data_df[f'price_t+{forward_return_period}'] = combined_data_df.groupby('stock_id')['price'].shift(-forward_return_period)
        combined_data_df['actual_forward_return'] = np.where( combined_data_df['price'] > 1e-9, (combined_data_df[f'price_t+{forward_return_period}'] / combined_data_df['price']) - 1, np.nan )
        combined_data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check required features exist
        missing_features_combined = [f for f in feature_names if f not in combined_data_df.columns];
        if missing_features_combined: raise ValueError(f"Combined data missing required features: {missing_features_combined}")

        # Handle NaNs in features BEFORE scaling/prediction
        feature_cols_combined = combined_data_df[feature_names].copy(); initial_feature_nan = feature_cols_combined.isnull().sum().sum()
        if initial_feature_nan > 0:
            logger.info(f"Found {initial_feature_nan} NaNs in combined features. Applying ffill/bfill per stock...")
            combined_data_df[feature_names] = combined_data_df.groupby('stock_id', group_keys=False)[feature_names].apply(lambda x: x.ffill().bfill())
            remaining_feature_nan = combined_data_df[feature_names].isnull().sum().sum()
            if remaining_feature_nan > 0: logger.warning(f"NaNs remain ({remaining_feature_nan}) after ffill/bfill in combined features. Rows with NaNs will not be predicted.")
        # Rows suitable for prediction are those with NO NaNs in features
        predict_mask = combined_data_df[feature_names].notna().all(axis=1); rows_to_predict_count = predict_mask.sum()
        if rows_to_predict_count == 0: raise ValueError("No rows with complete features available for prediction.")
        logger.info(f"Scaling and predicting on {rows_to_predict_count} rows with complete features.")

        X_combined_for_pred = combined_data_df.loc[predict_mask, feature_names].astype(np.float32)
        X_combined_scaled = scaler.transform(X_combined_for_pred)
        if np.isnan(X_combined_scaled).sum() > 0: raise ValueError("NaNs found after scaling combined features.")

        pred_batch_size = batch_size * 4 # Use larger batch for prediction speed
        combined_predictions = model.predict(X_combined_scaled, batch_size=pred_batch_size).flatten()
        logger.info(f"Predictions generated. Shape: {combined_predictions.shape}")

        # --- Step 5: Prepare Final Simulation DF ---
        logger.info("Preparing final dataframe for simulation...")
        # Select necessary columns
        extended_results_df = combined_data_df[['Date', 'stock_id', 'actual_forward_return', 'price']].copy()
        # Assign predictions only where features were available
        extended_results_df['predicted_score'] = np.nan # Initialize with NaN
        extended_results_df.loc[predict_mask, 'predicted_score'] = combined_predictions
        extended_results_df.rename(columns={'actual_forward_return': 'actual_return'}, inplace=True)

        # Drop rows where simulation cannot proceed (missing actual return or predicted score)
        rows_before_final_drop = len(extended_results_df)
        extended_results_df.dropna(subset=['actual_return', 'predicted_score'], inplace=True)
        logger.info(f"Dropped {rows_before_final_drop - len(extended_results_df)} rows with NaN actual_return or predicted_score.")

        if extended_results_df.empty: raise ValueError("Extended results DataFrame empty after final cleaning.")
        extended_results_df['Date'] = pd.to_datetime(extended_results_df['Date']).dt.normalize()
        extended_results_df = extended_results_df.set_index('Date').sort_index()
        final_sim_start_date = extended_results_df.index.min(); final_sim_end_date = extended_results_df.index.max()
        logger.info(f"Final simulation data ready. Shape: {extended_results_df.shape}. Dates: {final_sim_start_date.date()} to {final_sim_end_date.date()}")

        # --- Save intermediate DF for snapshot function (WORKAROUND) ---
        try:
             logger.info(f"Saving intermediate results DF for snapshot to {TEMP_EXTENDED_RESULTS_PATH}")
             extended_results_df.to_pickle(TEMP_EXTENDED_RESULTS_PATH)
        except Exception as save_err:
             logger.error(f"Could not save intermediate results DF: {save_err}")
             # Continue without saving, snapshot might fail later

        # --- Step 6: Run Extended Simulation ---
        logger.info(f"--- Running simulation: k={best_k}, n={best_n} ({final_sim_start_date.date()} to {final_sim_end_date.date()}) ---")
        extended_daily_returns, extended_metrics_base = run_simulation(
            extended_results_df.copy(), # Pass a copy to avoid modification issues
            best_k, best_n,
            period_name="Full Extended Period",
            risk_free_rate=risk_free,
            trading_days_year=trading_days_year
        )
        logger.info(f"Extended period simulation finished.")
        # --- Celery Update ---
        if task_context_ext:
             try:
                  step_start_progress = int(100 * (major_step_num - 1) / total_major_steps)
                  # Assume sim is 75% of this step's duration
                  overall_progress = step_start_progress + int(75 * (100 / total_major_steps))
                  task_context_ext.update_state(state='PROGRESS', meta={'step': 'Extended Simulation Done', 'progress': min(overall_progress, 95)})
             except Exception as e: logger.warning(f"Could not update Celery task state: {e}")

        # --- Step 7: Calculate Equity Curve ---
        dollar_equity_curve = None
        if extended_daily_returns is not None and not extended_daily_returns.empty:
            logger.info("Calculating dollar equity curve...")
            try:
                 sim_start_day_for_plot = extended_daily_returns.index.min() - pd.Timedelta(days=1);
                 start_capital_series = pd.Series([total_capital], index=[sim_start_day_for_plot]);
                 cumulative_factor = (1 + extended_daily_returns.fillna(0)).cumprod();
                 equity_curve_values = total_capital * cumulative_factor;
                 dollar_equity_curve = pd.concat([start_capital_series, equity_curve_values]).sort_index()
                 logger.info(f"Dollar equity curve calculated. End Value: {format_metric(dollar_equity_curve.iloc[-1], is_currency=True, currency_symbol=run_config.get('BASE_CURRENCY','$'))}")
            except Exception as eq_err:
                 logger.error(f"Error calculating equity curve: {eq_err}", exc_info=True)
                 dollar_equity_curve = None # Ensure it's None on error
        else: logger.warning("Could not calculate dollar equity curve: extended returns empty.")

        # --- Step 8: Calculate Comparison Metrics ---
        logger.info("Calculating comparison metrics for extended period...")
        extended_benchmark_daily_returns_aligned = pd.Series(dtype=float); extended_tracker_daily_returns_aligned = pd.Series(dtype=float)
        extended_benchmark_metrics = calculate_metrics(pd.Series(dtype=float)); extended_tracker_metrics = calculate_metrics(pd.Series(dtype=float)); comparison_metrics = calculate_benchmark_comparison_metrics(None, None)
        if extended_daily_returns is not None and not extended_daily_returns.empty:
             try: # Fetch comparison data logic...
                 bench_start_date = extended_daily_returns.index.min(); bench_end_date = extended_daily_returns.index.max() + pd.Timedelta(days=1)
                 tickers_to_fetch_ext = list(set([benchmark_ticker, tracker_fund_ticker])); logger.info(f"Fetching {tickers_to_fetch_ext} from {bench_start_date.date()} to {bench_end_date.date()}")
                 comp_data_ext = yf.download(tickers_to_fetch_ext, start=bench_start_date, end=bench_end_date, auto_adjust=True, progress=False, ignore_tz=True, timeout=30)
                 if not comp_data_ext.empty and comp_data_ext.shape[0] >= 2:
                      comp_data_ext.index = pd.to_datetime(comp_data_ext.index).normalize(); prices_ext_df = pd.DataFrame()
                      # Price extraction... (same logic as backtest_and_analyze)
                      if isinstance(comp_data_ext.columns, pd.MultiIndex) and 'Close' in comp_data_ext.columns.levels[0]: prices_ext_df = comp_data_ext['Close'].copy()
                      elif isinstance(comp_data_ext.columns, pd.Index):
                           present_tickers = [t for t in tickers_to_fetch_ext if t in comp_data_ext.columns];
                           if len(present_tickers) == len(tickers_to_fetch_ext): prices_ext_df = comp_data_ext[tickers_to_fetch_ext].copy()
                           elif 'Close' in comp_data_ext.columns:
                               if len(tickers_to_fetch_ext)==1: prices_ext_df=comp_data_ext[['Close']].rename(columns={'Close':tickers_to_fetch_ext[0]})
                               else: prices_ext_df=comp_data_ext[['Close']]; logger.warning("Flat comparison data, using 'Close'.")
                           else: logger.warning("Cannot find Close/ticker columns in flat comparison data.")
                      # Process prices_ext_df if extracted...
                      if not prices_ext_df.empty:
                           prices_ext_df = prices_ext_df.dropna(axis=1, how='all').dropna(axis=0, how='any')
                           if len(prices_ext_df) >= 2:
                                daily_returns_raw_ext = prices_ext_df.pct_change().dropna()
                                # Process Benchmark...
                                if benchmark_ticker in daily_returns_raw_ext.columns:
                                     bench_returns_raw_ext = daily_returns_raw_ext[benchmark_ticker]
                                     bench_returns_aligned = bench_returns_raw_ext.reindex(extended_daily_returns.index).dropna() # Align for metrics/comparison
                                     if not bench_returns_aligned.empty:
                                          extended_benchmark_metrics = calculate_metrics(bench_returns_aligned, risk_free, trading_days_year);
                                          extended_benchmark_daily_returns_aligned = bench_returns_aligned # Store aligned for return/plot
                                          comparison_metrics = calculate_benchmark_comparison_metrics(extended_daily_returns, extended_benchmark_daily_returns_aligned, risk_free, trading_days_year);
                                          logger.info("Benchmark base and comparison metrics calculated.")
                                     else: logger.warning(f"No benchmark returns after aligning to strategy dates.")
                                else: logger.warning(f"Benchmark ticker {benchmark_ticker} not found in downloaded comparison data.")
                                # Process Tracker...
                                if tracker_fund_ticker in daily_returns_raw_ext.columns:
                                     tracker_returns_raw_ext = daily_returns_raw_ext[tracker_fund_ticker]
                                     tracker_returns_aligned = tracker_returns_raw_ext.reindex(extended_daily_returns.index).dropna() # Align for metrics
                                     if not tracker_returns_aligned.empty:
                                          extended_tracker_metrics = calculate_metrics(tracker_returns_aligned, risk_free, trading_days_year);
                                          extended_tracker_daily_returns_aligned = tracker_returns_aligned # Store aligned for return/plot
                                          logger.info("Tracker base metrics calculated.")
                                     else: logger.warning(f"No tracker returns after aligning to strategy dates.")
                                else: logger.warning(f"Tracker ticker {tracker_fund_ticker} not found in downloaded comparison data.")
                           else: logger.warning("Not enough comparison price points after dropna.")
                      else: logger.warning("Could not extract prices from extended comparison data.")
                 else: logger.warning("Could not download sufficient comparison data for extended period.")
             except Exception as e: logger.error(f"Error calculating extended comparison metrics: {e}", exc_info=True)
        else: logger.warning("Skipping extended comparison metric calc because strategy returns are empty.")
        # Combine strategy base metrics + comparison metrics
        extended_metrics = {**extended_metrics_base, **comparison_metrics};
        # Explicitly add risk_free_rate used in calculations
        extended_metrics['risk_free_rate'] = risk_free

        # --- Step 9: Generate Plots ---
        logger.info("Generating extended period plots...")
        plot_paths = {} # Store relative paths of generated plots
        # --- Cumulative Plot ---
        if extended_daily_returns is not None and not extended_daily_returns.empty:
            try:
                fig, ax = plt.subplots(figsize=(16, 8)); extended_cumulative_returns=(1 + extended_daily_returns.fillna(0)).cumprod(); ax.plot(extended_cumulative_returns.index, extended_cumulative_returns, label=f'Strategy (k={best_k}, n={best_n})', lw=2.0, color='darkgreen', zorder=10)
                # Plot Benchmark (use aligned returns)
                if not extended_benchmark_daily_returns_aligned.empty:
                     bench_cumulative_ext = (1 + extended_benchmark_daily_returns_aligned.fillna(0)).cumprod();
                     # Plot only on intersecting dates for visual alignment
                     common_idx=extended_cumulative_returns.index.intersection(bench_cumulative_ext.index);
                     if not common_idx.empty: ax.plot(bench_cumulative_ext.loc[common_idx].index, bench_cumulative_ext.loc[common_idx], label=f'{benchmark_ticker} Benchmark', color='darkblue', linestyle='--', lw=1.5, zorder=5)
                     else: logger.warning("No overlapping dates to plot Benchmark vs Strategy.")
                # Plot Tracker (use aligned returns)
                if not extended_tracker_daily_returns_aligned.empty:
                     tracker_cumulative_ext = (1 + extended_tracker_daily_returns_aligned.fillna(0)).cumprod()
                     common_idx=extended_cumulative_returns.index.intersection(tracker_cumulative_ext.index);
                     if not common_idx.empty: ax.plot(tracker_cumulative_ext.loc[common_idx].index, tracker_cumulative_ext.loc[common_idx], label=f'{tracker_fund_ticker} Tracker', color='firebrick', linestyle=':', lw=1.5, zorder=4)
                     else: logger.warning("No overlapping dates to plot Tracker vs Strategy.")

                # Add shading/lines... (same logic as before using run_config dates)
                train_end_dt_cfg = pd.to_datetime(run_config.get('TRAIN_END_DATE')); val_end_dt_cfg = pd.to_datetime(run_config.get('VALIDATION_END_DATE')); plot_start_dt = extended_cumulative_returns.index.min(); plot_end_dt = extended_cumulative_returns.index.max()
                if pd.notna(train_end_dt_cfg) and train_end_dt_cfg >= plot_start_dt and train_end_dt_cfg <= plot_end_dt: ax.axvline(train_end_dt_cfg, color='grey', linestyle='--', lw=1.2, label='End Train')
                if pd.notna(val_end_dt_cfg) and val_end_dt_cfg >= plot_start_dt and val_end_dt_cfg <= plot_end_dt: ax.axvline(val_end_dt_cfg, color='purple', linestyle='--', lw=1.2, label='End Validation')
                if pd.notna(train_end_dt_cfg): ax.axvspan(plot_start_dt, min(train_end_dt_cfg, plot_end_dt), alpha=0.08, color='blue', label='Train')
                if pd.notna(train_end_dt_cfg) and pd.notna(val_end_dt_cfg): ax.axvspan(min(train_end_dt_cfg, plot_end_dt), min(val_end_dt_cfg, plot_end_dt), alpha=0.08, color='orange', label='Validation')
                if pd.notna(val_end_dt_cfg): ax.axvspan(min(val_end_dt_cfg, plot_end_dt), plot_end_dt, alpha=0.08, color='red', label='Test/Extended')
                # Formatting...
                start_str=plot_start_dt.strftime('%Y-%m-%d'); end_str=plot_end_dt.strftime('%Y-%m-%d'); ax.set_title(f'Strategy Cumulative Return (k={best_k}, n={best_n}) vs Comparisons\n{start_str} to {end_str}', fontsize=14); ax.set_ylabel('Cumulative Return (Log Scale)', fontsize=12); ax.set_xlabel('Date', fontsize=12); ax.set_yscale('log'); ax.legend(loc='upper left', fontsize=10); ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1, symbol='%')); ax.grid(True, which='both', linestyle='--', linewidth=0.5); plt.xticks(rotation=30, ha='right'); plt.tight_layout();
                # Save Plot
                full_plot_filename = f'dnn_backtest_EXTENDED_PERIOD_k{best_k}_n{best_n}_cumulative_return_log.png'; full_plot_path_final = os.path.join(output_dir, full_plot_filename); plt.savefig(full_plot_path_final, dpi=150); plt.close(fig); logger.info(f"Full period plot saved: {full_plot_path_final}"); plot_paths["full_period_plot_relpath"] = full_plot_filename
            except Exception as e: logger.error(f"Error plotting cumulative return: {e}", exc_info=True)
        else: logger.warning("Skipping cumulative return plot.")
        # --- Equity Curve Plot ---
        if dollar_equity_curve is not None and not dollar_equity_curve.empty:
             try: # Plot equity curve logic... save fig, close fig
                fig, ax = plt.subplots(figsize=(16, 8)); ax.plot(dollar_equity_curve.index, dollar_equity_curve, label=f'Strategy Equity (k={best_k}, n={best_n})', lw=2.0, color='darkgreen')
                # Add shading... (same logic as cumulative plot)
                train_end_dt_cfg = pd.to_datetime(run_config.get('TRAIN_END_DATE')); val_end_dt_cfg = pd.to_datetime(run_config.get('VALIDATION_END_DATE')); plot_start_dt = dollar_equity_curve.index.min(); plot_end_dt = dollar_equity_curve.index.max()
                if pd.notna(train_end_dt_cfg) and train_end_dt_cfg >= plot_start_dt and train_end_dt_cfg <= plot_end_dt: ax.axvline(train_end_dt_cfg, color='grey', linestyle='--', lw=1.2, label='End Train')
                if pd.notna(val_end_dt_cfg) and val_end_dt_cfg >= plot_start_dt and val_end_dt_cfg <= plot_end_dt: ax.axvline(val_end_dt_cfg, color='purple', linestyle='--', lw=1.2, label='End Validation')
                if pd.notna(train_end_dt_cfg): ax.axvspan(plot_start_dt, min(train_end_dt_cfg, plot_end_dt), alpha=0.08, color='blue')
                if pd.notna(train_end_dt_cfg) and pd.notna(val_end_dt_cfg): ax.axvspan(min(train_end_dt_cfg, plot_end_dt), min(val_end_dt_cfg, plot_end_dt), alpha=0.08, color='orange')
                if pd.notna(val_end_dt_cfg): ax.axvspan(min(val_end_dt_cfg, plot_end_dt), plot_end_dt, alpha=0.08, color='red')
                # Formatting...
                start_str = plot_start_dt.strftime('%Y-%m-%d'); end_str = plot_end_dt.strftime('%Y-%m-%d'); base_curr_sym = run_config.get('BASE_CURRENCY', '$');
                # Use dollar_formatter with the correct currency symbol
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: dollar_formatter(x, pos, currency_symbol=base_curr_sym)))
                ax.set_title(f'Strategy Dollar Equity Curve (k={best_k}, n={best_n})\n{start_str} to {end_str} (Start Capital: {format_metric(total_capital, is_currency=True, currency_symbol=base_curr_sym)})', fontsize=14); ax.set_ylabel(f'Portfolio Value ({base_curr_sym})', fontsize=12); ax.set_xlabel('Date', fontsize=12); ax.set_yscale('linear'); ax.legend(loc='upper left', fontsize=10); ax.grid(True, which='major', linestyle='--', lw=0.6); plt.xticks(rotation=30, ha='right'); plt.tight_layout();
                # Save Plot
                equity_plot_filename = f'dnn_backtest_EXTENDED_PERIOD_k{best_k}_n{best_n}_equity_curve.png'; equity_plot_path_final = os.path.join(output_dir, equity_plot_filename); plt.savefig(equity_plot_path_final, dpi=150); plt.close(fig); logger.info(f"Equity curve plot saved: {equity_plot_path_final}"); plot_paths["equity_curve_plot_relpath"] = equity_plot_filename
             except Exception as e: logger.error(f"Error plotting equity curve: {e}", exc_info=True)
        else: logger.warning("Skipping equity curve plot.")

        # --- Step 10: Prepare Lookback Data ---
        logger.info(f"Preparing lookback data for date: {lookback_date_str}...")
        lookback_info = {}
        try: # Lookback logic...
            target_date_lookback = pd.to_datetime(lookback_date_str).normalize();
            # Use the reloaded DF for lookback calculation if available, otherwise original
            df_for_lookback = extended_results_df
            if df_for_lookback.empty: raise ValueError("No results data available for lookback calculation.")

            sim_dates_lookback = df_for_lookback.index.unique().sort_values(); target_date_actual_ts = None
            # Find the actual simulation date >= target lookback date
            possible_dates_ts = sim_dates_lookback[sim_dates_lookback >= target_date_lookback]
            if not possible_dates_ts.empty: target_date_actual_ts = possible_dates_ts.min(); logger.info(f"Using actual sim date {target_date_actual_ts.date()} for lookback.")
            else: # If target is beyond sim range, use the last sim date
                if not sim_dates_lookback.empty: target_date_actual_ts = sim_dates_lookback.max(); logger.warning(f"Lookback date beyond sim range. Using last date {target_date_actual_ts.date()}.")
                else: raise ValueError("No sim dates available for lookback.")

            previous_trading_day_lookback_series = sim_dates_lookback[sim_dates_lookback < target_date_actual_ts]
            if previous_trading_day_lookback_series.empty: # Handle first day case
                logger.warning(f"Lookback: No sim day found before target date {target_date_actual_ts.date()}. Assuming start."); previous_trading_day_lookback_ts = None; holdings_start_lookback = set(); scores_prev_day_lookback = {}
            else: # Normal case
                 previous_trading_day_lookback_ts = previous_trading_day_lookback_series.max(); logger.info(f"Lookback Target: {target_date_actual_ts.date()}, Prev Day: {previous_trading_day_lookback_ts.date()}")
                 # Use get_holdings_on_date with the simulation results DF
                 holdings_start_lookback = get_holdings_on_date(df_for_lookback.copy(), best_k, best_n, target_date_actual_ts)
                 scores_prev_day_lookback = {};
                 if previous_trading_day_lookback_ts in df_for_lookback.index:
                     prev_day_slice = df_for_lookback.loc[[previous_trading_day_lookback_ts]];
                     if not prev_day_slice.empty:
                         # Ensure predicted_score exists and handle potential multi-index from .loc
                         if isinstance(prev_day_slice, pd.DataFrame):
                            df_slice_clean = prev_day_slice[['stock_id', 'predicted_score']].dropna()
                            scores_prev_day_lookback = pd.Series(df_slice_clean.predicted_score.values, index=df_slice_clean.stock_id).to_dict()
                         elif isinstance(prev_day_slice, pd.Series) and 'predicted_score' in prev_day_slice.index: # Handle Series case
                             scores_prev_day_lookback = {prev_day_slice['stock_id']: prev_day_slice['predicted_score']} if pd.notna(prev_day_slice['predicted_score']) else {}

                 else: logger.warning(f"Lookback: Prev day {previous_trading_day_lookback_ts.date()} not found for score lookup.")

            planned_buys_lookback, planned_sells_lookback = get_trades(scores_prev_day_lookback, holdings_start_lookback, best_k, best_n); holdings_end_lookback = (holdings_start_lookback - planned_sells_lookback).union(planned_buys_lookback)
            equity_start_lookback = np.nan; equity_end_lookback = np.nan
            if dollar_equity_curve is not None:
                 if previous_trading_day_lookback_ts is not None: equity_start_lookback = dollar_equity_curve.get(previous_trading_day_lookback_ts, np.nan)
                 else: equity_start_lookback = total_capital # Assume start of simulation
                 equity_end_lookback = dollar_equity_curve.get(target_date_actual_ts, np.nan)

            # Store dates as objects for now, convert later
            lookback_info = {
                "target_date": target_date_actual_ts.to_pydatetime().date(), # Store as date object
                "prev_date": previous_trading_day_lookback_ts.to_pydatetime().date() if previous_trading_day_lookback_ts else None,
                "holdings_start": sorted(list(holdings_start_lookback)),
                "scores_prev_day": scores_prev_day_lookback,
                "planned_sells": sorted(list(planned_sells_lookback)),
                "planned_buys": sorted(list(planned_buys_lookback)),
                "holdings_end": sorted(list(holdings_end_lookback)),
                "equity_start": equity_start_lookback,
                "equity_end": equity_end_lookback
            };
            logger.info(f"Lookback data prepared for {target_date_actual_ts.date()}.")
        except ValueError as ve: logger.error(f"Error during lookback prep: {ve}")
        except Exception as e: logger.error(f"Unexpected lookback prep error: {e}", exc_info=True)

        # --- Step 11: Prepare Next Day Data ---
        logger.info("Preparing next day estimation data...")
        next_day_info = {}
        try: # Next day logic...
            # Use the reloaded DF if available, otherwise original
            df_for_next_day = extended_results_df
            if not df_for_next_day.empty:
                 sim_dates_next = df_for_next_day.index.unique().sort_values()
                 if not sim_dates_next.empty:
                     most_recent_sim_date_ts = sim_dates_next[-1]; logger.info(f"Estimating next day based on data up to: {most_recent_sim_date_ts.date()}")
                     # Target date for holdings is the day *after* the last simulation day
                     next_day_target_for_holdings_ts = most_recent_sim_date_ts + pd.Timedelta(days=1)
                     # Get holdings at the START of that next day
                     holdings_start_next_day = get_holdings_on_date(df_for_next_day.copy(), best_k, best_n, next_day_target_for_holdings_ts)

                     scores_end_most_recent = {};
                     if most_recent_sim_date_ts in df_for_next_day.index:
                          today_data_slice = df_for_next_day.loc[[most_recent_sim_date_ts]]
                          if not today_data_slice.empty:
                               if isinstance(today_data_slice, pd.DataFrame):
                                    df_slice_clean = today_data_slice[['stock_id', 'predicted_score']].dropna(); scores_end_most_recent = pd.Series(df_slice_clean.predicted_score.values, index=df_slice_clean.stock_id).to_dict()
                               elif isinstance(today_data_slice, pd.Series): # Handle Series
                                    scores_end_most_recent = {today_data_slice['stock_id']: today_data_slice['predicted_score']} if pd.notna(today_data_slice['predicted_score']) else {}
                     else: logger.warning(f"Next Day: Most recent sim date {most_recent_sim_date_ts.date()} not found for score lookup.")

                     planned_buys_next, planned_sells_next = get_trades(scores_end_most_recent, holdings_start_next_day, best_k, best_n);
                     estimated_holdings_end_next = (holdings_start_next_day - planned_sells_next).union(planned_buys_next);
                     equity_start_next = np.nan
                     if dollar_equity_curve is not None: equity_start_next = dollar_equity_curve.get(most_recent_sim_date_ts, np.nan)

                     # Store dates as objects for now
                     next_day_info = {
                         "based_on_date": most_recent_sim_date_ts.to_pydatetime().date(), # Store as date object
                         "holdings_start": sorted(list(holdings_start_next_day)),
                         "scores_current": scores_end_most_recent,
                         "planned_sells": sorted(list(planned_sells_next)),
                         "planned_buys": sorted(list(planned_buys_next)),
                         "estimated_holdings_end": sorted(list(estimated_holdings_end_next)),
                         "equity_start": equity_start_next
                     };
                     logger.info(f"Next day estimation data prepared.")
                 else: logger.warning("No sim data dates found for next day estimate.")
            else: logger.warning("Extended results df empty, cannot estimate next day.")
        except Exception as e: logger.error(f"Error during next day prep: {e}", exc_info=True)

        # --- Step 12: Final Return Bundle ---
        logger.info("Preparing final result bundle...")

        # Helper to convert Timestamps in dict keys/values to ISO strings safely
        def stringify_dict_elements(d):
             if not isinstance(d, dict): return d
             new_dict = {}
             for k, v in d.items():
                  key_str = k.isoformat() if isinstance(k, (pd.Timestamp, datetime)) else str(k)
                  val_str = v.isoformat() if isinstance(v, (pd.Timestamp, datetime)) else v
                  # Handle nested dicts recursively (optional, only if needed)
                  # if isinstance(val_str, dict): val_str = stringify_dict_keys(val_str)
                  new_dict[key_str] = val_str
             return new_dict

        # Convert relevant dates/series to JSON serializable types
        final_sim_start_str = final_sim_start_date.isoformat() if pd.notna(final_sim_start_date) else None
        final_sim_end_str = final_sim_end_date.isoformat() if pd.notna(final_sim_end_date) else None
        if lookback_info.get('target_date'): lookback_info['target_date'] = lookback_info['target_date'].isoformat()
        if lookback_info.get('prev_date'): lookback_info['prev_date'] = lookback_info['prev_date'].isoformat()
        if next_day_info.get('based_on_date'): next_day_info['based_on_date'] = next_day_info['based_on_date'].isoformat()

        # Convert Series/DataFrames to dicts *before* returning
        # Use .to_dict('index') or similar if index is important, otherwise just values
        extended_daily_returns_dict = stringify_dict_elements(extended_daily_returns.to_dict()) if extended_daily_returns is not None else None
        bench_returns_dict = stringify_dict_elements(extended_benchmark_daily_returns_aligned.to_dict()) if not extended_benchmark_daily_returns_aligned.empty else None
        tracker_returns_dict = stringify_dict_elements(extended_tracker_daily_returns_aligned.to_dict()) if not extended_tracker_daily_returns_aligned.empty else None
        equity_curve_dict = stringify_dict_elements(dollar_equity_curve.to_dict()) if dollar_equity_curve is not None else None

        # Clean final metrics dicts of non-serializable types (just in case)
        def clean_metrics(metrics_dict):
            cleaned = {}
            for k, v in metrics_dict.items():
                if isinstance(v, (np.integer)): cleaned[k] = int(v)
                elif isinstance(v, (np.floating)): cleaned[k] = float(v) if np.isfinite(v) else None
                elif isinstance(v, (np.bool_)): cleaned[k] = bool(v)
                elif pd.isna(v): cleaned[k] = None
                else: cleaned[k] = v # Keep other types like str, int, float
            return cleaned

        cleaned_extended_metrics = clean_metrics(extended_metrics)
        cleaned_bench_metrics = clean_metrics(extended_benchmark_metrics)
        cleaned_tracker_metrics = clean_metrics(extended_tracker_metrics)

        return {
            "strategy_name": run_config.get('STRATEGY_NAME'),
            "base_currency": run_config.get('BASE_CURRENCY'),
            "best_k": best_k, "best_n": best_n,
            "extended_daily_returns": extended_daily_returns_dict, # Now dict
            "extended_metrics": cleaned_extended_metrics,
            "extended_benchmark_daily_returns": bench_returns_dict, # Now dict
            "extended_benchmark_metrics": cleaned_bench_metrics,
            "extended_tracker_daily_returns": tracker_returns_dict, # Now dict
            "extended_tracker_metrics": cleaned_tracker_metrics,
            "dollar_equity_curve": equity_curve_dict, # Now dict
            "total_capital": total_capital,
            "lookback_info": lookback_info, # Dates are now strings
            "next_day_info": next_day_info, # Dates are now strings
            "benchmark_ticker": benchmark_ticker,
            "tracker_fund_ticker": tracker_fund_ticker,
            "full_period_start_date": final_sim_start_str, # String
            "full_period_end_date": final_sim_end_str, # String
            "plot_paths": plot_paths # Dict containing relative plot paths
            # Removed "extended_results_df" from final return payload
        }

    except Exception as e:
        logger.error(f"Error during extended sim/reporting prep: {e}", exc_info=True)
        raise # Re-raise to be caught by main workflow handler
    finally:
        # Clean up temp file regardless of success/failure if it exists
        if os.path.exists(TEMP_EXTENDED_RESULTS_PATH):
            try: os.remove(TEMP_EXTENDED_RESULTS_PATH); logger.info("Removed temporary snapshot data file.")
            except OSError as rm_err: logger.error(f"Error removing temp snapshot file: {rm_err}")

# === END SECTION 6 ===


# === SECTION 7: Report Generation ===
# (generate_text_report is IMPORTED from report_utils)

# === SECTION 8: CSV Metrics Saving ===
# (Keep save_metrics_to_csv as adjusted previously)
def save_metrics_to_csv(report_data, output_csv_path):
    """Extracts key metrics and saves to CSV."""
    # ... (implementation as provided in previous response) ...
    logger.info(f"--- Saving Key Metrics to CSV: {output_csv_path} ---")
    try:
        strat_metrics = report_data.get('extended_metrics', {}); bench_metrics = report_data.get('extended_benchmark_metrics', {}); tracker_metrics = report_data.get('extended_tracker_metrics', {})
        best_k = report_data.get('best_k', 'N/A'); best_n = report_data.get('best_n', 'N/A')
        start_str = report_data.get('full_period_start_date', "N/A"); end_str = report_data.get('full_period_end_date', "N/A")
        if isinstance(start_str, str): start_str = start_str[:10] # Ensure only date part if ISO string
        if isinstance(end_str, str): end_str = end_str[:10]     # Ensure only date part if ISO string
        benchmark_ticker = report_data.get('benchmark_ticker', 'N/A'); tracker_fund_ticker = report_data.get('tracker_fund_ticker', 'N/A'); strategy_name = report_data.get('strategy_name', 'N/A')
        metrics_to_save = ['Cumulative Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Positive Days %', 'Trading Days', 'Beta', 'Alpha (Jensen)', 'Correlation', 'Tracking Error']
        bench_col_name = f'Benchmark ({benchmark_ticker})' if benchmark_ticker != 'N/A' else 'Benchmark'
        tracker_col_name = f'Tracker ({tracker_fund_ticker})' if tracker_fund_ticker != 'N/A' else 'Tracker'
        data_for_csv = { 'Metric': [], 'Strategy': [], bench_col_name: [], tracker_col_name: [] }
        data_for_csv['Metric'].extend(['Strategy_Name', 'Parameter_K', 'Parameter_N', 'Period_Start', 'Period_End'])
        data_for_csv['Strategy'].extend([strategy_name, best_k, best_n, start_str, end_str])
        data_for_csv[bench_col_name].extend(['N/A', 'N/A', 'N/A', start_str, end_str])
        data_for_csv[tracker_col_name].extend(['N/A', 'N/A', 'N/A', start_str, end_str])
        for metric in metrics_to_save:
            metric_key = metric.split(' (')[0]; # Handle base key for Sharpe Ratio etc.
            if metric == 'Alpha (Jensen)': metric_key = 'Alpha (Jensen)' # Use full key if needed
            strat_value = strat_metrics.get(metric_key); bench_value = bench_metrics.get(metric_key); tracker_value = tracker_metrics.get(metric_key)
            data_for_csv['Metric'].append(metric); data_for_csv['Strategy'].append(strat_value)
            if metric_key in ['Beta', 'Alpha (Jensen)', 'Correlation', 'Tracking Error']:
                 if metric_key == 'Beta': bench_value_display = 1.0
                 elif metric_key == 'Alpha (Jensen)': bench_value_display = 0.0
                 elif metric_key == 'Correlation': bench_value_display = 1.0
                 elif metric_key == 'Tracking Error': bench_value_display = 0.0
                 else: bench_value_display = None
                 tracker_value_display = None
            else: bench_value_display = bench_value; tracker_value_display = tracker_value
            data_for_csv[bench_col_name].append(bench_value_display); data_for_csv[tracker_col_name].append(tracker_value_display)
        metrics_df = pd.DataFrame(data_for_csv); metrics_df.to_csv(output_csv_path, index=False, float_format='%.6f', na_rep='N/A')
        logger.info(f"Metrics successfully saved to {output_csv_path}")
    except Exception as e: logger.error(f"Failed to save metrics to CSV: {e}", exc_info=True)

# === SECTION 9: Portfolio Snapshot CSV Saving ===
# (Keep save_portfolio_snapshot_to_csv as adjusted previously)
def save_portfolio_snapshot_to_csv(
    snapshot_date, # datetime.date object expected here
    holdings_start_day, # Set of tickers held at START of snapshot_date
    holdings_end_day, # Set of tickers held at END of snapshot_date
    full_results_df, # DataFrame with index=Date, columns='stock_id', 'actual_return', 'predicted_score', 'price'
    output_csv_path,
    top_n_movers=5
    ):
    """Generates portfolio snapshot CSV."""
    # ... (implementation as provided in previous response) ...
    logger.info(f"--- Generating Portfolio Snapshot CSV for {snapshot_date}: {output_csv_path} ---")
    try:
        snapshot_data_list = []
        try: snapshot_ts = pd.to_datetime(snapshot_date).normalize() # Convert date obj to Timestamp
        except Exception as e: logger.error(f"Invalid snapshot_date: {snapshot_date}. Error: {e}."); return
        if full_results_df is None or full_results_df.empty: logger.warning("Snapshot: Full results DF empty."); return
        if not isinstance(full_results_df.index, pd.DatetimeIndex): logger.error("Snapshot: full_results_df needs DatetimeIndex."); return

        working_df = full_results_df.copy() # Work on a copy
        required_cols_full = ['stock_id', 'predicted_score', 'actual_return', 'price']
        for col in required_cols_full:
            if col not in working_df.columns: logger.warning(f"Snapshot: DF missing '{col}'."); working_df[col] = np.nan
        working_df['actual_return'] = pd.to_numeric(working_df['actual_return'], errors='coerce')
        working_df['predicted_score'] = pd.to_numeric(working_df['predicted_score'], errors='coerce')
        working_df['price'] = pd.to_numeric(working_df['price'], errors='coerce')
        working_df['stock_id'] = working_df['stock_id'].astype(str) if 'stock_id' in working_df else None
        working_df = working_df.dropna(subset=['stock_id'])

        results_up_to_snapshot = working_df[working_df.index <= snapshot_ts].copy()
        cumulative_returns = pd.Series(dtype=float)
        if not results_up_to_snapshot.empty:
            logger.info(f"Calculating cumulative returns up to {snapshot_date}...")
            results_up_to_snapshot['cumul_factor'] = 1 + results_up_to_snapshot['actual_return'].fillna(0)
            cumulative_returns = results_up_to_snapshot.groupby('stock_id')['cumul_factor'].prod() - 1
            logger.info(f"Calculated cumulative returns for {len(cumulative_returns)} stocks.")
        else: logger.warning(f"No data found up to snapshot date {snapshot_date}.")

        daily_info_map = {}
        if snapshot_ts in results_up_to_snapshot.index:
             daily_data_slice = results_up_to_snapshot.loc[[snapshot_ts]] # Keep as potentially multi-index DF
             # Convert to dict keyed by stock_id
             if not daily_data_slice.empty:
                daily_info_map = daily_data_slice.reset_index().set_index('stock_id').to_dict('index')
        else: logger.warning(f"Snapshot date {snapshot_date} not found in results index.")

        held_start_list = list(holdings_start_day)
        cumulative_returns_held_start = cumulative_returns.reindex(held_start_list).dropna()

        def add_snapshot_row(stock_id, category, rank=None):
             daily_info = daily_info_map.get(stock_id, {}); cumul_ret = cumulative_returns.get(stock_id, np.nan)
             snapshot_data_list.append({
                 'Date': snapshot_date.isoformat(), # Store date as ISO string
                 'Category': category,
                 'Rank': rank, # Will be None for current holdings
                 'StockID': stock_id,
                 'Cumulative Return': cumul_ret if pd.notna(cumul_ret) else None,
                 'Snapshot Date Price': daily_info.get('price') if pd.notna(daily_info.get('price')) else None,
                 'Snapshot Date Daily Return': daily_info.get('actual_return') if pd.notna(daily_info.get('actual_return')) else None,
                 'Snapshot Date Score': daily_info.get('predicted_score') if pd.notna(daily_info.get('predicted_score')) else None
             })

        if not cumulative_returns_held_start.empty:
            top_gainers_series = cumulative_returns_held_start.nlargest(top_n_movers)
            processed_for_movers = set()
            for rank, (stock_id, _) in enumerate(top_gainers_series.items(), 1):
                add_snapshot_row(stock_id, 'Top Gainer (Held Start, Cumul)', rank); processed_for_movers.add(stock_id)
            top_losers_series = cumulative_returns_held_start.nsmallest(top_n_movers)
            rank_loser = 0
            for stock_id, _ in top_losers_series.items():
                 if stock_id not in processed_for_movers: # Avoid listing same stock as gain/loss
                      rank_loser += 1
                      add_snapshot_row(stock_id, 'Top Loser (Held Start, Cumul)', rank_loser); processed_for_movers.add(stock_id)
        else: logger.info(f"No stocks held at start of {snapshot_date} with returns for movers.")

        holdings_end_list = sorted(list(holdings_end_day))
        if holdings_end_list:
            logger.info(f"Adding {len(holdings_end_list)} holdings from end of {snapshot_date}.")
            for stock_id in holdings_end_list:
                add_snapshot_row(stock_id, 'Current Holding (End Day)', None) # Rank is None
        else: logger.info(f"No holdings at end of {snapshot_date}.")

        if snapshot_data_list:
            snapshot_df = pd.DataFrame(snapshot_data_list)
            final_cols = ['Date', 'Category', 'Rank', 'StockID', 'Cumulative Return', 'Snapshot Date Price', 'Snapshot Date Daily Return', 'Snapshot Date Score']
            for col in final_cols: # Ensure all columns exist
                if col not in snapshot_df.columns: snapshot_df[col] = None
            snapshot_df = snapshot_df[final_cols] # Reorder
            snapshot_df['Category'] = pd.Categorical( snapshot_df['Category'], categories=['Top Gainer (Held Start, Cumul)', 'Top Loser (Held Start, Cumul)', 'Current Holding (End Day)'], ordered=True )
            snapshot_df.sort_values(by=['Date', 'Category', 'Rank'], inplace=True, na_position='last')
            snapshot_df.to_csv(output_csv_path, index=False, float_format='%.6f', na_rep='N/A')
            logger.info(f"Portfolio snapshot successfully saved to {output_csv_path}")
        else: logger.warning(f"No data for portfolio snapshot on {snapshot_date}. CSV not created.")
    except Exception as e: logger.error(f"Failed to save portfolio snapshot CSV for {snapshot_date}: {e}", exc_info=True)
# === END SECTION 9 ===


# === MAIN WORKFLOW FUNCTION ===
def run_full_strategy_workflow(config: dict, output_base_dir: str):
    """
    Runs the entire strategy workflow. `config` should be the merged dict.
    """
    run_config = config # Use the merged config passed from the task
    strategy_name = run_config.get('STRATEGY_NAME', 'Unnamed Run')
    run_id = os.path.basename(output_base_dir)
    logger.info(f"===== Starting Workflow: {strategy_name} (Run ID: {run_id}) =====")
    logger.info(f"Output directory: {output_base_dir}")

    set_seeds(run_config['SEED'])

    # Define file paths relative to the unique run output directory
    RAW_DATA_OUTPUT_CSV = os.path.join(output_base_dir, "detailed_data_with_alphas_and_price.csv")
    PROCESSED_DATA_DIR = os.path.join(output_base_dir, "processed_dnn_data_regression")
    MODEL_SAVE_FILENAME = 'dnn_alpha_yield_model.keras'
    MODEL_SAVE_PATH = os.path.join(output_base_dir, MODEL_SAVE_FILENAME)
    TRAINING_HISTORY_PLOT = os.path.join(output_base_dir, 'dnn_training_history.png')
    REPORT_FILE = os.path.join(output_base_dir, 'strategy_performance_report.txt')
    METRICS_CSV_FILE = os.path.join(output_base_dir, 'strategy_performance_metrics.csv')
    PORTFOLIO_SNAPSHOT_CSV_FILE = os.path.join(output_base_dir, 'strategy_portfolio_snapshot.csv')
    TEMP_EXTENDED_RESULTS_PATH = os.path.join(output_base_dir, "temp_extended_results_for_snapshot.pkl") # Temp file path

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) # Ensure needed dirs exist

    # --- Progress Reporting Setup ---
    task = current_task._get_current_object() if hasattr(current_task, '_get_current_object') else None
    total_major_steps = 6 # Define total major steps

    def update_progress_local(step_name, current_step, total_steps):
        """Helper function to update Celery task progress (now active)."""
        progress = 0
        if total_steps > 0: progress = int(100 * current_step / total_steps)
        logger.info(f"--- Progress Step {current_step}/{total_steps}: {step_name} ({progress}%) ---")
        if task:
            try:
                # Only send step name and overall progress
                task.update_state(state='PROGRESS', meta={'step': step_name, 'progress': progress})
            except Exception as e:
                logger.warning(f"Could not update Celery task state: {e}")
        pass # End of update_progress_local

    # --- Main Workflow Steps ---
    report_data_final = {}
    extended_results_df_local_copy = pd.DataFrame() # For snapshot workaround
    best_k = None; best_n = None; test_plot_relpath = None; final_result_payload = {}

    try:
        # Step 1: Fetch & Process Raw Data
        update_progress_local("Fetch & Process Raw Data", 1, total_major_steps)
        logger.info("\n----- STEP 1: Fetching and Processing Raw Data -----")
        filtered_data_csv_path = fetch_and_process_raw_data(
             tickers=run_config['TICKERS'], start_date_str=run_config['START_DATE_STR'],
             end_date_str=run_config['VALIDATION_END_DATE'], # Use validation end for initial fetch
             benchmark_ticker=run_config['BENCHMARK_TICKER'],
             selected_factors=run_config['SELECTED_ALPHA_COLS_PHASE_II'],
             output_csv_path=RAW_DATA_OUTPUT_CSV
        )

        # Step 2: Preprocess Data
        update_progress_local("Preprocess Data", 2, total_major_steps)
        logger.info("\n----- STEP 2: Preprocessing Data for DNN -----")
        scaler_path, feature_list_path = preprocess_data_for_dnn(
             input_csv_path=filtered_data_csv_path, output_dir=PROCESSED_DATA_DIR,
             alpha_cols_list=run_config['SELECTED_ALPHA_COLS_PHASE_II'],
             forward_return_period=run_config['FORWARD_RETURN_PERIOD'],
             train_end_dt_str=run_config['TRAIN_END_DATE'], val_end_dt_str=run_config['VALIDATION_END_DATE']
        )

        # Step 3: Train Model
        update_progress_local("Train DNN Model", 3, total_major_steps)
        logger.info("\n----- STEP 3: Training DNN Model -----")
        trained_model_path = train_dnn_model(
             processed_data_dir=PROCESSED_DATA_DIR, model_save_path=MODEL_SAVE_PATH,
             feature_list_path=feature_list_path, plot_save_path=TRAINING_HISTORY_PLOT,
             hidden_units=run_config['HIDDEN_UNITS'], hidden_activation=run_config['HIDDEN_ACTIVATION'],
             output_activation=run_config['OUTPUT_ACTIVATION'], learning_rate=run_config['LEARNING_RATE'],
             epochs=run_config['EPOCHS'], patience=run_config['PATIENCE'], batch_size=run_config['BATCH_SIZE'],
             dropout_rate=run_config['DROPOUT_RATE'], seed=run_config['SEED'],
             major_step_num=3, total_major_steps=total_major_steps # Pass progress info
        )

        # Step 4: Backtest & Grid Search
        update_progress_local("Backtest & Grid Search", 4, total_major_steps)
        logger.info("\n----- STEP 4: Backtesting and Grid Search -----")
        backtest_results = backtest_and_analyze(
             processed_data_dir=PROCESSED_DATA_DIR, model_path=trained_model_path,
             scaler_path=scaler_path, feature_list_path=feature_list_path,
             train_end_date_str=run_config['TRAIN_END_DATE'], validation_end_date_str=run_config['VALIDATION_END_DATE'],
             # Use the correct config keys for K/N candidates
             k_candidates_list=run_config['K_VALUES_CANDIDATES'],
             n_candidates_list=run_config['N_VALUES_CANDIDATES'],
             validation_metric=run_config['VALIDATION_OPTIMIZATION_METRIC'], n_to_test=run_config['N_CANDIDATES_TO_TEST'],
             benchmark_ticker=run_config['BENCHMARK_TICKER'], tracker_fund_ticker=run_config['TRACKER_FUND_TICKER'],
             risk_free=run_config['RISK_FREE_RATE_ANNUAL'], trading_days_year=run_config['TRADING_DAYS_PER_YEAR'],
             forward_return_period=run_config['FORWARD_RETURN_PERIOD'], batch_size=run_config['BATCH_SIZE'],
             output_dir=output_base_dir, seed=run_config['SEED'],
             major_step_num=4, total_major_steps=total_major_steps # Pass progress info
        )
        best_k = backtest_results['best_k']; best_n = backtest_results['best_n']
        test_plot_relpath = backtest_results.get("test_period_plot_relpath")
        logger.info(f"Best parameters from Validation Grid Search: k={best_k}, n={best_n}")

        # Step 5: Extended Simulation & Report Prep
        update_progress_local("Extended Simulation", 5, total_major_steps)
        logger.info("\n----- STEP 5: Extended Simulation and Report Data Preparation -----")
        # Pass run_config down
        report_data_final = run_extended_simulation_and_prep_report_data(
            original_filtered_data_csv=filtered_data_csv_path,
            feature_names=run_config['SELECTED_ALPHA_COLS_PHASE_II'], tickers=run_config['TICKERS'],
            model_path=trained_model_path, scaler_path=scaler_path,
            best_k=best_k, best_n=best_n,
            forward_return_period=run_config['FORWARD_RETURN_PERIOD'],
            # Pass necessary args directly from run_config
            extend_end_date_str=run_config['EXTEND_SIMULATION_END_DATE_STR'],
            total_capital=run_config['TOTAL_STRATEGY_CAPITAL'],
            benchmark_ticker=run_config['BENCHMARK_TICKER'],
            tracker_fund_ticker=run_config['TRACKER_FUND_TICKER'],
            risk_free=run_config['RISK_FREE_RATE_ANNUAL'],
            trading_days_year=run_config['TRADING_DAYS_PER_YEAR'],
            lookback_date_str=run_config['LOOKBACK_DATE_STR'],
            batch_size=run_config['BATCH_SIZE'], output_dir=output_base_dir, seed=run_config['SEED'],
            major_step_num=5, total_major_steps=total_major_steps,
            run_config=run_config # Pass the full config dict
        )
        # report_data_final now contains the JSON-serializable results

        # --- Load results DF for snapshot (WORKAROUND) ---
        if os.path.exists(TEMP_EXTENDED_RESULTS_PATH):
             try:
                 extended_results_df_local_copy = pd.read_pickle(TEMP_EXTENDED_RESULTS_PATH)
                 # Ensure index is datetime after loading from pickle
                 if not isinstance(extended_results_df_local_copy.index, pd.DatetimeIndex):
                      logger.info("Converting pickled index to DatetimeIndex for snapshot...")
                      extended_results_df_local_copy.index = pd.to_datetime(extended_results_df_local_copy.index)
                 logger.info(f"Reloaded intermediate results DF for snapshot. Shape: {extended_results_df_local_copy.shape}")
             except Exception as load_err:
                  logger.error(f"Error reloading intermediate results DF: {load_err}")
                  extended_results_df_local_copy = pd.DataFrame() # Ensure it's a DF on error
        else:
             logger.warning(f"Temp file not found: {TEMP_EXTENDED_RESULTS_PATH}. Snapshot generation might fail.")
             extended_results_df_local_copy = pd.DataFrame() # Ensure it's defined

        # Step 6: Generate Reports & Save Files
        update_progress_local("Generate Reports & Save Files", 6, total_major_steps)
        logger.info("\n----- STEP 6: Generating Final Performance Report & Saving Artifacts -----")

        # Text Report
        if report_data_final and 'strategy_name' in report_data_final:
            generate_text_report(report_data_final, REPORT_FILE)
        else: logger.error("Cannot generate text report: report_data incomplete.")

        # Metrics CSV
        if report_data_final and 'strategy_name' in report_data_final:
            save_metrics_to_csv(report_data_final, METRICS_CSV_FILE)
        else: logger.warning("Skipping metrics CSV saving.")

 # Portfolio Snapshot CSV
        lookback_info = report_data_final.get('lookback_info', {})
        # Get the date ISO STRING from lookback_info (it was converted for JSON)
        actual_snapshot_date_str = lookback_info.get('target_date')

        # Convert ISO string back to date object for the check and function call
        actual_snapshot_date_parsed = None
        if isinstance(actual_snapshot_date_str, str):
             try:
                 actual_snapshot_date_parsed = datetime.fromisoformat(actual_snapshot_date_str).date()
             except ValueError:
                 logger.error(f"Could not parse snapshot target date string: {actual_snapshot_date_str}")

        snapshot_possible = (
            actual_snapshot_date_parsed is not None and
            # **** FIX: Use the imported 'date' type directly ****
            isinstance(actual_snapshot_date_parsed, date) and
            # *****************************************************
            isinstance(extended_results_df_local_copy, pd.DataFrame) and
            not extended_results_df_local_copy.empty and
            best_k is not None
        )
        if snapshot_possible:
             logger.info("\n--- Saving Final Portfolio Snapshot CSV ---")
             save_portfolio_snapshot_to_csv(
                 snapshot_date=actual_snapshot_date_parsed, # Pass the actual date object
                 holdings_start_day=set(lookback_info.get('holdings_start', [])),
                 holdings_end_day=set(lookback_info.get('holdings_end', [])),
                 full_results_df=extended_results_df_local_copy, # Use reloaded DF
                 output_csv_path=PORTFOLIO_SNAPSHOT_CSV_FILE
             )
        else:
            logger.warning("Skipping portfolio snapshot CSV saving - conditions not met.")
            if actual_snapshot_date_parsed is None:
                logger.warning(f"  Reason: Could not parse snapshot date ('{actual_snapshot_date_str}')")
            elif not isinstance(extended_results_df_local_copy, pd.DataFrame) or extended_results_df_local_copy.empty:
                 logger.warning(f"  Reason: Snapshot source data frame is invalid or empty.")
            elif best_k is None:
                 logger.warning(f"  Reason: Best K parameter is missing.")


        # --- Final Success ---
        logger.info(f"\n===== Strategy Workflow Completed Successfully: {strategy_name} (Run ID: {run_id}) =====")

        # Prepare final result payload for Celery/API
        # Import FileInfo model locally to avoid potential circular imports if structure changes
        from ..models.workflow import FileInfo
        plot_paths_dict = report_data_final.get("plot_paths", {}) # Get plot paths dict from report data
        # Create FileInfo payload, checking existence for robustness
        files_payload = FileInfo(
             metrics_csv=os.path.basename(METRICS_CSV_FILE) if os.path.exists(METRICS_CSV_FILE) else None,
             snapshot_csv=os.path.basename(PORTFOLIO_SNAPSHOT_CSV_FILE) if os.path.exists(PORTFOLIO_SNAPSHOT_CSV_FILE) else None,
             report_txt=os.path.basename(REPORT_FILE) if os.path.exists(REPORT_FILE) else None,
             training_history_plot=os.path.basename(TRAINING_HISTORY_PLOT) if os.path.exists(TRAINING_HISTORY_PLOT) else None,
             full_period_plot=plot_paths_dict.get("full_period_plot_relpath"), # Get path from dict
             equity_curve_plot=plot_paths_dict.get("equity_curve_plot_relpath"), # Get path from dict
             test_period_plot=test_plot_relpath, # Get from backtest results
             log_file=f"{run_id}.log" # Log file name matches run id
        )
        # Construct the final dictionary to be returned by the Celery task
        final_result_payload = {
            "message": "Workflow completed successfully.",
            "run_id": run_id,
            "best_k": best_k,
            "best_n": best_n,
            # Convert FileInfo model instance to dict for JSON serialization
            "files": files_payload.model_dump(exclude_none=True)
        }
        return final_result_payload # Return the success payload

    # --- Main Exception Handling ---
    except Exception as e:
        logger.critical(f"!!! Workflow FAILED Unhandled in run_full_strategy_workflow: {e} (Run ID: {run_id}) !!!", exc_info=True)
        # Ensure temp file cleanup happens even on failure
        if os.path.exists(TEMP_EXTENDED_RESULTS_PATH):
             try: os.remove(TEMP_EXTENDED_RESULTS_PATH); logger.info("Removed temporary snapshot data file on error.")
             except OSError as rm_err: logger.error(f"Error removing temp snapshot file on error: {rm_err}")
        # Re-raise the exception so Celery marks the task as FAILED
        # The tasks.py wrapper will handle creating the failure payload
        raise e
    finally:
         # Clean up temp file on success only if it still exists
         # The check `final_result_payload.get("message")` is removed as it might not be set if error occurred before final step
         if os.path.exists(TEMP_EXTENDED_RESULTS_PATH):
             try:
                 # Check if we are exiting due to success (no exception raised)
                 # This is tricky without explicit success flag, but presence of file implies it was created
                 logger.info("Attempting to remove temporary snapshot data file in finally block.")
                 os.remove(TEMP_EXTENDED_RESULTS_PATH)
             except NameError: # final_result_payload might not exist if error happened early
                 pass
             except OSError as rm_err:
                 logger.error(f"Error removing temp snapshot file in finally block: {rm_err}")

# === END MAIN WORKFLOW FUNCTION ===
