# backend/app/utils/report_utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date # Keep date import
import logging
import os

# Import formatting utilities
try:
    from .formatting_utils import format_metric, get_sorted_scores
except ImportError:
    logging.error("CRITICAL: formatting_utils.py not found or cannot be imported.")
    def format_metric(value, **kwargs): return str(value) if value is not None else "N/A"
    def get_sorted_scores(stock_set, score_dict, **kwargs): return [(s, score_dict.get(s, 'N/A')) for s in stock_set]

logger = logging.getLogger(__name__)

def generate_text_report(report_data, output_file):
    """Generates a text-based strategy performance report including Tracker Fund."""
    logger.info(f"--- Generating Text Performance Report to {output_file} ---")
    rep = []
    sep = "=" * 85
    ssep = "-" * 85

    # --- Extract data using .get() with defaults ---
    strategy_name = report_data.get('strategy_name', 'N/A')
    base_currency = report_data.get('base_currency', '$')
    benchmark_ticker = report_data.get('benchmark_ticker', 'N/A')
    tracker_fund_ticker = report_data.get('tracker_fund_ticker', 'N/A')
    best_k = report_data.get('best_k', 'N/A')
    best_n = report_data.get('best_n', 'N/A')

    # --- FIX: Dates are already strings from report_data ---
    start_dt_str = report_data.get('full_period_start_date') # Get the string
    end_dt_str = report_data.get('full_period_end_date')     # Get the string

    # Extract only the date part (YYYY-MM-DD) if it's an ISO string
    start_str_display = start_dt_str[:10] if isinstance(start_dt_str, str) else "N/A"
    end_str_display = end_dt_str[:10] if isinstance(end_dt_str, str) else "N/A"
    # --- END FIX ---

    risk_free = 0.0
    extended_metrics_data = report_data.get('extended_metrics', {})
    if isinstance(extended_metrics_data, dict):
         try:
             rf_value = extended_metrics_data.get('risk_free_rate', 0.0)
             risk_free = float(rf_value) if rf_value is not None else 0.0
         except (ValueError, TypeError):
             risk_free = 0.0

    # --- Report Header ---
    rep.append(sep)
    rep.append(f" Strategy Performance Report: {strategy_name}")
    rep.append(sep)
    rep.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rep.append(f" Base Currency: {base_currency}")
    # Use the display strings
    rep.append(f" Full Analysis Period: {start_str_display} to {end_str_display}")
    rep.append(f" Benchmark: {benchmark_ticker}")
    rep.append(f" Tracker Fund: {tracker_fund_ticker}")
    rep.append(f" Strategy Parameters (Top Validation): k={best_k}, n={best_n}")
    rep.append(ssep)
    rep.append(" Key Performance Indicators (Full Extended Period)")
    rep.append(ssep)

    # --- Extract Metrics ---
    strat_metrics = report_data.get('extended_metrics', {})
    bench_metrics = report_data.get('extended_benchmark_metrics', {})
    tracker_metrics = report_data.get('extended_tracker_metrics', {})

    # --- Metrics Table ---
    col_width = 18; strat_header = 'Strategy'; bench_header = f'Bench ({benchmark_ticker})'; tracker_header = f'Tracker ({tracker_fund_ticker})'
    header_fmt = f"   {{:<28}} | {{:<{col_width}}} | {{:<{col_width}}} | {{:<{col_width}}}"; row_fmt = header_fmt; separator = f"   {'-'*28}-+-{'-'*col_width}-+-{'-'*col_width}-+-{'-'*col_width}"
    rep.append(header_fmt.format('Metric', strat_header, bench_header, tracker_header)); rep.append(separator)
    metrics_to_display_base = [('Cumulative Return', True, 4), ('Annualized Return', True, 4), ('Annualized Volatility', True, 4), (f'Sharpe Ratio (Rf={risk_free*100:.1f}%)', False, 3, 'Sharpe Ratio'), ('Sortino Ratio', False, 3), ('Max Drawdown', True, 4), ('Positive Days %', False, 2, 'Positive Days %', True), ('Trading Days', False, 0)]
    for metric_display_info in metrics_to_display_base:
        metric_name = metric_display_info[0]; is_perc_style = metric_display_info[1]; precision = metric_display_info[2]; metric_key = metric_display_info[3] if len(metric_display_info) > 3 else metric_name.split(' (')[0]; manual_percent_sign = metric_display_info[4] if len(metric_display_info) > 4 else False
        strat_val = strat_metrics.get(metric_key); bench_val = bench_metrics.get(metric_key); tracker_val = tracker_metrics.get(metric_key)
        strat_formatted = format_metric(strat_val, precision=precision, is_percent=is_perc_style, currency_symbol=base_currency); bench_formatted = format_metric(bench_val, precision=precision, is_percent=is_perc_style, currency_symbol=base_currency); tracker_formatted = format_metric(tracker_val, precision=precision, is_percent=is_perc_style, currency_symbol=base_currency)
        if manual_percent_sign and strat_formatted != "N/A": strat_formatted += '%';
        if manual_percent_sign and bench_formatted != "N/A": bench_formatted += '%';
        if manual_percent_sign and tracker_formatted != "N/A": tracker_formatted += '%'
        rep.append(row_fmt.format(metric_name, strat_formatted, bench_formatted, tracker_formatted))
    rep.append(separator)
    metrics_to_display_comp = [('Beta (vs Benchmark)', False, 4, 'Beta'), ('Alpha (Jensen, ann.)', True, 4, 'Alpha (Jensen)'), ('Correlation (vs Bench)', False, 4, 'Correlation'), ('Tracking Error (ann.)', True, 4, 'Tracking Error')]
    for metric_name, is_perc_style, precision, metric_key in metrics_to_display_comp:
         bench_val_disp = "N/A";
         if metric_key == 'Beta': bench_val_disp = format_metric(1.0, precision=precision)
         elif metric_key == 'Alpha (Jensen)': bench_val_disp = format_metric(0.0, precision=precision, is_percent=True)
         elif metric_key == 'Correlation': bench_val_disp = format_metric(1.0, precision=precision)
         elif metric_key == 'Tracking Error': bench_val_disp = format_metric(0.0, precision=precision, is_percent=True)
         strat_comp_val = strat_metrics.get(metric_key); strat_comp_formatted = format_metric(strat_comp_val, precision=precision, is_percent=is_perc_style, currency_symbol=base_currency)
         rep.append(row_fmt.format(metric_name, strat_comp_formatted, bench_val_disp, "N/A"))
    rep.append(ssep)

    # --- Equity Curve Summary ---
    equity_curve_dict = report_data.get('dollar_equity_curve'); rep.append(" Equity Curve Summary"); rep.append(ssep)
    if equity_curve_dict and isinstance(equity_curve_dict, dict) and len(equity_curve_dict) > 1:
        equity_values = []; equity_dates = []
        try: # Try sorting assuming keys are ISO date strings
             sorted_items = sorted(equity_curve_dict.items(), key=lambda item: datetime.fromisoformat(item[0]))
             equity_dates = [item[0] for item in sorted_items]
             equity_values = [item[1] for item in sorted_items]
        except ValueError: # Fallback if keys aren't sortable dates
             logger.warning("Equity curve dictionary keys could not be sorted as dates. Using original order.")
             equity_dates = list(equity_curve_dict.keys())
             equity_values = list(equity_curve_dict.values())

        start_cap = equity_values[0]; end_equity = equity_values[-1];
        equity_start_date_str = equity_dates[0][:10]; # Use first key's date part
        equity_end_date_str = equity_dates[-1][:10]; # Use last key's date part
        profit = end_equity - start_cap if pd.notna(start_cap) and pd.notna(end_equity) else np.nan;
        peak_equity = max(equity_values) if equity_values else np.nan;
        lowest_equity = min(equity_values) if equity_values else np.nan;
        cur = base_currency + ' '
        rep.append(f"   Starting Capital ({equity_start_date_str}): {format_metric(start_cap, is_currency=True, currency_symbol=cur)}");
        rep.append(f"   Ending Equity ({equity_end_date_str}):    {format_metric(end_equity, is_currency=True, currency_symbol=cur)}");
        rep.append(f"   Total P/L:                 {format_metric(profit, is_currency=True, currency_symbol=cur)}");
        rep.append(f"   Peak Equity:               {format_metric(peak_equity, is_currency=True, currency_symbol=cur)}");
        rep.append(f"   Lowest Equity:             {format_metric(lowest_equity, is_currency=True, currency_symbol=cur)}")
    else: rep.append("   Equity Curve Data Not Available or Insufficient.")
    rep.append(ssep)


    # --- Lookback Analysis ---
    lookback = report_data.get('lookback_info', {}); rep.append(" Lookback Analysis"); rep.append(ssep)
    # Dates in lookback_info are already ISO strings after run_extended_... finishes
    target_date_str_look = lookback.get('target_date')
    if lookback and target_date_str_look:
        target_date_display = target_date_str_look[:10] # Get date part
        prev_date_str_look = lookback.get('prev_date', 'N/A');
        if prev_date_str_look and prev_date_str_look != 'N/A':
             prev_date_display = prev_date_str_look[:10] # Get date part
        else: prev_date_display = 'N/A'

        rep.append(f" Analysis for Date: {target_date_display}");
        eq_start = lookback.get('equity_start'); eq_end = lookback.get('equity_end'); holdings_start = lookback.get('holdings_start', []); planned_sells = lookback.get('planned_sells', []); planned_buys = lookback.get('planned_buys', []); holdings_end = lookback.get('holdings_end', []); scores_prev = lookback.get('scores_prev_day', {}); cur = base_currency + ' '
        rep.append(f"  Holdings at Start ({len(holdings_start)} stocks, entering {target_date_display}):");
        rep.append(f"    Equity (Close {prev_date_display}): {format_metric(eq_start, is_currency=True, currency_symbol=cur)}");
        rep.append(f"    Tickers: {', '.join(holdings_start) if holdings_start else 'None'}");
        rep.append(f"  Planned Sells ({len(planned_sells)} stocks, on {target_date_display}): {', '.join(planned_sells) if planned_sells else 'None'}")
        sell_scores = get_sorted_scores(planned_sells, scores_prev, reverse_sort=False, limit=5);
        if sell_scores: rep.append(f"    Scores ({prev_date_display}, low->high): {', '.join([f'{s}:{format_metric(v, 5)}' for s, v in sell_scores])}")
        rep.append(f"  Planned Buys ({len(planned_buys)} stocks, on {target_date_display}): {', '.join(planned_buys) if planned_buys else 'None'}")
        buy_scores = get_sorted_scores(planned_buys, scores_prev, reverse_sort=True, limit=5);
        if buy_scores: rep.append(f"    Scores ({prev_date_display}, high->low): {', '.join([f'{s}:{format_metric(v, 5)}' for s, v in buy_scores])}")
        rep.append(f"  Holdings at End ({len(holdings_end)} stocks, close {target_date_display}):");
        rep.append(f"    Equity (Close {target_date_display}): {format_metric(eq_end, is_currency=True, currency_symbol=cur)}");
        rep.append(f"    Tickers: {', '.join(holdings_end) if holdings_end else 'None'}")
    else: rep.append("   Lookback Analysis Data Not Available or Invalid Date.")
    rep.append(ssep)


    # --- Next Day Estimation ---
    next_day = report_data.get('next_day_info', {}); rep.append(" Next Day Estimation"); rep.append(ssep)
    # Date is already ISO string
    based_on_date_str_next = next_day.get('based_on_date')
    if next_day and based_on_date_str_next:
        based_on_date_display = based_on_date_str_next[:10]
        next_trading_day_est_str = "N/A"
        try: # Estimate next day string
             next_trading_day_est_str = (datetime.fromisoformat(based_on_date_str_next) + timedelta(days=1)).strftime('%Y-%m-%d')
        except (ValueError, TypeError): pass
        next_day_display = f"~{next_trading_day_est_str}" if next_trading_day_est_str != "N/A" else "Next Trading Day"

        rep.append(f" Trades for {next_day_display}, based on {based_on_date_display} scores:");
        eq_start_next = next_day.get('equity_start'); holdings_start_next = next_day.get('holdings_start', []); planned_sells_next = next_day.get('planned_sells', []); planned_buys_next = next_day.get('planned_buys', []); estimated_holdings_end = next_day.get('estimated_holdings_end', []); scores_curr = next_day.get('scores_current', {}); cur = base_currency + ' '
        rep.append(f"  Holdings at Start ({len(holdings_start_next)} stocks, entering {next_day_display}):");
        rep.append(f"    Based on Equity (Close {based_on_date_display}): {format_metric(eq_start_next, is_currency=True, currency_symbol=cur)}");
        rep.append(f"    Tickers: {', '.join(holdings_start_next) if holdings_start_next else 'None'}");
        rep.append(f"  PLANNED Sells for Next Day ({len(planned_sells_next)} stocks): {', '.join(planned_sells_next) if planned_sells_next else 'None'}")
        sell_scores_next = get_sorted_scores(planned_sells_next, scores_curr, reverse_sort=False, limit=5);
        if sell_scores_next: rep.append(f"    Scores ({based_on_date_display}, low->high): {', '.join([f'{s}:{format_metric(v, 5)}' for s, v in sell_scores_next])}")
        rep.append(f"  PLANNED Buys for Next Day ({len(planned_buys_next)} stocks): {', '.join(planned_buys_next) if planned_buys_next else 'None'}")
        buy_scores_next = get_sorted_scores(planned_buys_next, scores_curr, reverse_sort=True, limit=5);
        if buy_scores_next: rep.append(f"    Scores ({based_on_date_display}, high->low): {', '.join([f'{s}:{format_metric(v, 5)}' for s, v in buy_scores_next])}")
        rep.append(f"  Estimated Holdings at END of Next Day ({len(estimated_holdings_end)} stocks):");
        rep.append(f"    Tickers: {', '.join(estimated_holdings_end) if estimated_holdings_end else 'None'}")
    else: rep.append("   Next Day Estimation Data Not Available.")
    rep.append(ssep)

    # --- Disclaimer ---
    rep.append(" Disclaimer:")
    rep.append("   Performance metrics based on simulated trading. Past performance is not indicative of future results.")
    rep.append("   Simulations exclude transaction costs, slippage, taxes, and liquidity constraints.")
    rep.append("   This report is for informational purposes only. All investment decisions involve risk.")
    rep.append(sep)

    # --- Write Report ---
    try:
        # Ensure output directory exists before writing
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(rep))
        logger.info(f"Text report successfully written to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write report file '{output_file}': {e}", exc_info=True)