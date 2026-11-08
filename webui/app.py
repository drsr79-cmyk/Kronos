import os
import sys
import json
import logging
import secrets
import datetime
import warnings

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

warnings.filterwarnings('ignore')
logger = logging.getLogger("kronos.webui")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kronos_safety import (
    FORECAST_LIMITATION,
    SafetyError,
    enforce_inference_budget,
    limit,
    list_data_files,
    loopback_addr,
    resolve_local_data_file,
    scrub_text,
    secrets_equal,
    text_contains_secret,
    name_looks_secret,
    validate_device,
    webui_runtime_settings,
    install_log_redactor,
)

install_log_redactor(logger)

try:
    import plotly.graph_objects as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    logger.warning("Kronos model library is not importable in this process")

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
_UI_TOKEN = None

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'local_model': 'Kronos-mini',
        'local_tokenizer': 'Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'local_model': 'Kronos-small',
        'local_tokenizer': 'Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'local_model': 'Kronos-base',
        'local_tokenizer': 'Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def ui_token() -> str:
    global _UI_TOKEN
    env_token = os.environ.get("KRONOS_WEBUI_TOKEN", "").strip()
    if env_token:
        if len(env_token) < 16:
            raise SafetyError("weak_token", "KRONOS_WEBUI_TOKEN must be at least 16 characters")
        return env_token
    if _UI_TOKEN is None:
        _UI_TOKEN = secrets.token_urlsafe(32)
    return _UI_TOKEN


def request_authorized() -> bool:
    if request.headers.get("X-Kronos-Request") != "1":
        return False
    token = ui_token()
    header_token = request.headers.get("X-Kronos-Token", "")
    cookie_token = request.cookies.get("kronos_session", "")
    if loopback_addr(request.remote_addr):
        return secrets_equal(header_token, token) or secrets_equal(cookie_token, token)
    return secrets_equal(header_token, token)


def public_error(exc: Exception, status: int = 400):
    if isinstance(exc, SafetyError):
        logger.info("rejected request code=%s", exc.code)
        return jsonify({"error": exc.code, "limitation": FORECAST_LIMITATION}), status
    logger.warning("request failed: %s", exc.__class__.__name__)
    return jsonify({"error": "request_failed", "limitation": FORECAST_LIMITATION}), status


@app.errorhandler(Exception)
def hide_internal_errors(exc):
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, SafetyError):
        return public_error(exc)
    logger.warning("unhandled request error: %s", exc.__class__.__name__)
    return jsonify({"error": "request_failed", "limitation": FORECAST_LIMITATION}), 500


@app.before_request
def guard_request():
    addr = request.remote_addr or ""
    nonlocal_ok = os.environ.get("KRONOS_ALLOW_NONLOCAL_BIND") == "1" and len(os.environ.get("KRONOS_WEBUI_TOKEN", "")) >= 16
    if not loopback_addr(addr) and not nonlocal_ok:
        return jsonify({"error": "forbidden"}), 403
    if request.path == "/" and request.method == "GET":
        return None
    if request.path.startswith("/api/"):
        try:
            if not request_authorized():
                return jsonify({"error": "unauthorized"}), 401
        except SafetyError as exc:
            return public_error(exc, 403)
        return None
    return jsonify({"error": "not_found"}), 404


@app.after_request
def harden_response(response):
    if request.path == "/" and request.method == "GET" and response.status_code == 200:
        try:
            response.set_cookie(
                "kronos_session",
                ui_token(),
                httponly=True,
                samesite="Strict",
                secure=False,
                path="/",
                max_age=8 * 60 * 60,
            )
        except SafetyError:
            logger.warning("UI token was rejected")
    for header in (
        "Access-Control-Allow-Origin",
        "Access-Control-Allow-Credentials",
        "Access-Control-Allow-Private-Network",
        "Access-Control-Allow-Headers",
        "Access-Control-Allow-Methods",
    ):
        response.headers.pop(header, None)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    )
    return response


def load_data_files():
    """Scan the confined data directory and return file names only."""
    try:
        return list_data_files()
    except SafetyError:
        return []

def load_data_file(file_path):
    """Load a CSV or Feather file from the confined data directory."""
    try:
        resolved = resolve_local_data_file(file_path)
        max_rows = limit("max_rows")
        if resolved.suffix.lower() == ".csv":
            df = pd.read_csv(resolved, nrows=max_rows + 1)
        elif resolved.suffix.lower() == ".feather":
            df = pd.read_feather(resolved)
        else:
            raise SafetyError("unsupported_type", "only .csv and .feather files are allowed")
        if len(df) > max_rows:
            raise SafetyError("too_many_rows", "data file exceeds the engineering row limit")
        secret_columns = [column for column in df.columns if text_contains_secret(str(column)) or name_looks_secret(str(column))]
        if secret_columns:
            df = df.drop(columns=secret_columns)
        for column in list(df.columns):
            if df[column].dtype == object:
                mask = df[column].astype(str).map(text_contains_secret)
                if mask.any():
                    df.loc[mask, column] = pd.NA
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, SafetyError("invalid_columns", "data file is missing required price columns")
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None

    except SafetyError as exc:
        return None, exc
    except Exception:
        logger.warning("data file could not be parsed")
        return None, SafetyError("unreadable_data", "data file could not be read")

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file after scrubbing secret-like strings from the payload.
        rendered = scrub_text(json.dumps(save_data, ensure_ascii=True))
        if text_contains_secret(rendered):
            logger.warning("prediction export dropped because it still contained secret-like material")
            return None
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(rendered)
        
        logger.info("prediction results saved")
        return filepath
        
    except Exception:
        logger.warning("prediction results were not saved")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart when Plotly is installed. The browser does not load a CDN."""
    if not PLOTLY_AVAILABLE:
        return None
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return public_error(SafetyError("invalid_json", "invalid request"))
        file_path = data.get('file_path')
        
        if not file_path:
            return public_error(SafetyError("invalid_data_path", "invalid data path"))
        
        df, error = load_data_file(file_path)
        if error:
            return public_error(error if isinstance(error, SafetyError) else SafetyError("unreadable_data", "data file could not be read"))
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows',
            'limitation': FORECAST_LIMITATION,
        })
        
    except SafetyError as exc:
        return public_error(exc)
    except Exception as exc:
        return public_error(exc)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return public_error(SafetyError("invalid_json", "invalid request"))
        file_path = data.get('file_path')
        try:
            budget = enforce_inference_budget(
                lookback=data.get('lookback', 400),
                pred_len=data.get('pred_len', 120),
                sample_count=data.get('sample_count', 1),
                temperature=data.get('temperature', 1.0),
                top_p=data.get('top_p', 0.9),
            )
        except SafetyError as exc:
            return public_error(exc)
        lookback = budget["lookback"]
        pred_len = budget["pred_len"]
        temperature = budget["temperature"]
        top_p = budget["top_p"]
        sample_count = budget["sample_count"]
        
        if not file_path:
            return public_error(SafetyError("invalid_data_path", "invalid data path"))
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return public_error(error if isinstance(error, SafetyError) else SafetyError("unreadable_data", "data file could not be read"))
        
        if len(df) < lookback:
            return public_error(SafetyError("insufficient_rows", "not enough rows for the requested lookback"))
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return public_error(SafetyError("insufficient_rows", "not enough rows in the selected window"))
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                logger.warning("prediction failed: %s", e.__class__.__name__)
                return public_error(SafetyError("prediction_failed", "prediction failed"))
        else:
            return public_error(SafetyError("model_not_loaded", "load a local model before predicting"))
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # Prediction uses first 400 data points
            # Actual data should be 120 data points after first 400 data points
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: start from beginning
            historical_start_idx = 0
        
        chart_json = None
        if PLOTLY_AVAILABLE:
            try:
                chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
            except Exception:
                logger.warning("chart generation skipped")
                chart_json = None
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date:
                # Custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                # Latest data: calculate from last time point of entire data file
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else ''),
            'limitation': FORECAST_LIMITATION,
        })
        
    except SafetyError as exc:
        return public_error(exc)
    except Exception as exc:
        return public_error(exc)

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return public_error(SafetyError("invalid_json", "invalid request"))
        model_key = data.get('model_key', 'kronos-small')
        device = validate_device(data.get('device', 'cpu'))
        if not MODEL_AVAILABLE:
            return public_error(SafetyError("model_unavailable", "model library is not available"))
        
        if model_key not in AVAILABLE_MODELS:
            return public_error(SafetyError("unknown_model", "unknown model"))
        
        model_config = AVAILABLE_MODELS[model_key]
        model_root = os.environ.get("KRONOS_MODEL_DIR", os.path.join(_ROOT, "pretrained"))
        if os.environ.get("KRONOS_ALLOW_HUB_DOWNLOAD") == "1":
            tokenizer_source = model_config['tokenizer_id']
            model_source = model_config['model_id']
        else:
            tokenizer_source = os.path.join(model_root, model_config['local_tokenizer'])
            model_source = os.path.join(model_root, model_config['local_model'])

        tokenizer = KronosTokenizer.from_pretrained(tokenizer_source)
        model = Kronos.from_pretrained(model_source)
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            },
            'limitation': FORECAST_LIMITATION,
        })
        
    except SafetyError as exc:
        return public_error(exc)
    except Exception as exc:
        return public_error(exc)

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

def main():
    settings = webui_runtime_settings()
    shown_host = "127.0.0.1" if settings["host"] in {"127.0.0.1", "::1"} else settings["host"]
    print("Starting Kronos Web UI")
    print(f"Open http://{shown_host}:{settings['port']} from this machine")
    print(FORECAST_LIMITATION)
    print("The debugger is disabled. Hub download stays off unless KRONOS_ALLOW_HUB_DOWNLOAD=1 was set before startup.")
    app.run(
        host=settings["host"],
        port=settings["port"],
        debug=False,
        use_reloader=False,
        load_dotenv=False,
    )


if __name__ == '__main__':
    main()
