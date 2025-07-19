# backend/app/utils/formatting_utils.py
import pandas as pd
import numpy as np
import logging

# Define a default currency symbol within this utility file.
# The actual symbol used can be overridden by passing the `currency_symbol` argument.
DEFAULT_CURRENCY_SYMBOL = 'HKD' # Set a sensible default (e.g., from your strategy's base currency)

logger = logging.getLogger(__name__)

# --- format_metric function ---
def format_metric(value, precision=4, is_percent=False, is_currency=False, currency_symbol=None):
    """Formats metrics for display, handling None/NaN/Inf."""
    if value is None or pd.isna(value) or (isinstance(value, (float, np.number)) and not np.isfinite(value)):
        return "N/A"
    try:
        val_float = float(value)
        # Use default currency symbol if not provided and currency formatting is requested
        symbol_to_use = currency_symbol if currency_symbol is not None else DEFAULT_CURRENCY_SYMBOL

        if is_percent:
            # Ensure precision is valid for percentage formatting (e.g., precision=4 means .2%)
            display_precision = max(0, precision - 2) if precision >= 2 else 0
            return f"{val_float:.{display_precision}%}"
        elif is_currency:
            # Use 0 decimal places for currency in reports/summaries for cleaner look
            return f"{symbol_to_use}{val_float:,.0f}"
        else:
            # Standard float formatting with specified precision
            return f"{val_float:.{precision}f}"
    except (ValueError, TypeError):
        logger.warning(f"Could not format metric value: {value}")
        return "N/A" # Return N/A if formatting fails

# --- get_sorted_scores function ---
def get_sorted_scores(stock_set, score_dict, reverse_sort=True, limit=5):
    """Sorts a given set of stocks based on their scores from a dictionary."""
    if not stock_set or not score_dict:
        return [] # Return empty list if no stocks or scores

    stock_set_str = {str(s) for s in stock_set} # Ensure stock IDs are strings
    items = []
    for stock in stock_set_str:
        # Get the score and attempt to convert to float, default to NaN on failure
        raw_score = score_dict.get(stock)
        score = np.nan # Default to NaN
        if pd.notna(raw_score):
            try:
                score = float(raw_score)
            except (ValueError, TypeError):
                 logger.debug(f"Could not convert score '{raw_score}' for stock '{stock}' to float. Treating as NaN.")
                 score = np.nan # Ensure it's NaN if conversion fails
        items.append((stock, score))

    # Define sort key function to handle NaNs correctly
    # NaNs should always be sorted last, regardless of ascending/descending order
    def sort_key(item):
        score_value = item[1]
        is_nan = pd.isna(score_value)
        # Assign a very large/small number for NaNs to push them to the end
        nan_placeholder = float('inf') # NaNs go last in ascending sort
        if reverse_sort:
            nan_placeholder = float('-inf') # NaNs go last in descending sort (using negative infinity)

        # Return a tuple: (is_nan_flag, actual_score_or_placeholder)
        # This ensures NaNs group together at the end based on the is_nan_flag (False sorts before True)
        return (is_nan, score_value if not is_nan else nan_placeholder)

    try:
        # Sort the items list using the custom key
        items.sort(key=sort_key, reverse=reverse_sort)
    except Exception as sort_err:
        logger.error(f"Error sorting scores: {sort_err}", exc_info=True)
        return [] # Return empty list on unexpected sorting error

    # Return the sorted list, limited by the 'limit' parameter
    return items[:limit] if limit and limit > 0 else items


# --- dollar_formatter function (for matplotlib compatibility) ---
def dollar_formatter(x, pos=None, currency_symbol=None):
    """
    Formats a number as currency for matplotlib tick labels.
    Handles NaN/Inf and ensures no decimals for cleaner axes.
    'pos' argument is required by matplotlib FuncFormatter but often unused.
    """
    # Use default currency symbol if not provided
    symbol_to_use = currency_symbol if currency_symbol is not None else DEFAULT_CURRENCY_SYMBOL

    # Check for invalid numeric inputs first
    if pd.isna(x) or (isinstance(x, (float, np.number)) and not np.isfinite(x)):
        return "N/A" # Return N/A for NaN or Inf

    try:
        # Format as currency with commas and no decimal places
        return f'{symbol_to_use}{float(x):,.0f}'
    except (ValueError, TypeError):
        # If conversion to float fails, return N/A
        return "N/A"

# Ensure the duplicated functions below are REMOVED
# def format_metric(...): <-- REMOVE THIS DUPLICATE
# def get_sorted_scores(...): <-- REMOVE THIS DUPLICATE