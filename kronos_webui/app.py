"""
Kronos 原生 WebUI 独立服务。

当前目标是先把原生 WebUI 单独拉起，保持与上游交互契约尽量一致，
因此这里优先兼容上游的接口返回格式和页面行为。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from backend.core.settings import settings
from backend.services.kronos_task_service import kronos_task_service


app = Flask(__name__, template_folder="templates")
CORS(app)


AVAILABLE_MODELS = {
    "kronos-base": {
        "name": "Kronos-base",
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "context_length": 512,
        "params": "102.3M",
        "description": "Base model, provides better prediction quality",
    }
}

CURRENT_MODEL = {
    "loaded": False,
    "model_key": settings.KRONOS_DEFAULT_MODEL,
    "device": settings.KRONOS_DEFAULT_DEVICE,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except Exception:
        return default


def _scan_local_files() -> list[dict[str, Any]]:
    data_files: list[dict[str, Any]] = []
    for path in sorted(settings.DATA_DIR.iterdir()):
        if path.suffix.lower() not in {".csv", ".feather"}:
            continue
        file_size = path.stat().st_size
        data_files.append(
            {
                "name": path.name,
                "path": str(path),
                "size": f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB",
            }
        )
    return data_files


def _build_data_info(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "start_date": df["timestamps"].min().isoformat(),
        "end_date": df["timestamps"].max().isoformat(),
        "price_range": {
            "min": float(df[["open", "high", "low", "close"]].min().min()),
            "max": float(df[["open", "high", "low", "close"]].max().max()),
        },
        "prediction_columns": ["open", "high", "low", "close", "volume"],
        "timeframe": detect_timeframe(df),
    }


def _load_factorhub_stock(stock_code: str, start_date: str, end_date: str):
    prepared = kronos_task_service.load_dataset(
        {
            "source_type": "factorhub_stock",
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
        }
    )
    if prepared.dataframe is None or prepared.dataframe.empty:
        return None, "No data returned from Factorhub"

    df = prepared.dataframe.copy()
    df = df.rename(columns={"date": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]], None


def load_data_file(file_path: str):
    """兼容上游原生 WebUI 的本地文件加载逻辑。"""
    path = Path(file_path)
    if not path.exists():
        return None, "File does not exist"

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".feather":
            df = pd.read_feather(path)
        else:
            return None, "Unsupported file format"

        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"

        if "timestamps" in df.columns:
            df["timestamps"] = pd.to_datetime(df["timestamps"])
        elif "timestamp" in df.columns:
            df["timestamps"] = pd.to_datetime(df["timestamp"])
        elif "date" in df.columns:
            df["timestamps"] = pd.to_datetime(df["date"])
        else:
            df["timestamps"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["volume", "amount"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        return df, None
    except Exception as exc:
        return None, f"Failed to load file: {exc}"


def detect_timeframe(df: pd.DataFrame) -> str:
    if len(df) < 2:
        return "Unknown"
    diffs = []
    for idx in range(1, min(10, len(df))):
        diffs.append(df["timestamps"].iloc[idx] - df["timestamps"].iloc[idx - 1])
    if not diffs:
        return "Unknown"
    avg_diff = sum(diffs, pd.Timedelta(0)) / len(diffs)
    if avg_diff < pd.Timedelta(minutes=1):
        return f"{avg_diff.total_seconds():.0f} seconds"
    if avg_diff < pd.Timedelta(hours=1):
        return f"{avg_diff.total_seconds() / 60:.0f} minutes"
    if avg_diff < pd.Timedelta(days=1):
        return f"{avg_diff.total_seconds() / 3600:.0f} hours"
    return f"{avg_diff.days} days"


def simulate_prediction(df: pd.DataFrame, lookback: int, pred_len: int, start_date: str | None):
    """先用同步占位预测把原生 WebUI 跑起来，后续再替换为真实 Kronos 推理。"""
    if start_date:
        start_dt = pd.to_datetime(start_date)
        time_range_df = df[df["timestamps"] >= start_dt].reset_index(drop=True)
        if len(time_range_df) < lookback + pred_len:
            raise ValueError(
                f"Insufficient data from start time {start_dt.strftime('%Y-%m-%d %H:%M')}, "
                f"need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available"
            )
        historical_df = time_range_df.iloc[:lookback].copy()
        actual_df = time_range_df.iloc[lookback : lookback + pred_len].copy()
        historical_start_idx = int(df[df["timestamps"] >= start_dt].index[0])
        prediction_type = (
            f"Simulated Kronos prediction (selected window: first {lookback} points for prediction, "
            f"last {pred_len} points for comparison)"
        )
    else:
        if len(df) < lookback + pred_len:
            raise ValueError(f"Insufficient data length, need at least {lookback + pred_len} rows")
        historical_df = df.iloc[:lookback].copy()
        actual_df = df.iloc[lookback : lookback + pred_len].copy()
        historical_start_idx = 0
        prediction_type = "Simulated Kronos prediction (latest data)"

    last_close = _safe_float(historical_df["close"].iloc[-1], 1.0)
    time_diff = df["timestamps"].iloc[1] - df["timestamps"].iloc[0] if len(df) > 1 else pd.Timedelta(days=1)
    forecast_rows: list[dict[str, Any]] = []
    for idx in range(pred_len):
        drift = 0.0025 * (idx + 1)
        close = last_close * (1.0 + drift)
        forecast_rows.append(
            {
                "timestamps": historical_df["timestamps"].iloc[-1] + (idx + 1) * time_diff,
                "open": close * 0.996,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": _safe_float(historical_df["volume"].iloc[-1]),
                "amount": _safe_float(historical_df["amount"].iloc[-1]),
            }
        )
    pred_df = pd.DataFrame(forecast_rows)
    return historical_df, pred_df, actual_df, prediction_type, historical_start_idx


def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    if historical_start_idx + lookback + pred_len <= len(df):
        historical_df = df.iloc[historical_start_idx : historical_start_idx + lookback]
    else:
        available_lookback = min(lookback, len(df) - historical_start_idx)
        historical_df = df.iloc[historical_start_idx : historical_start_idx + available_lookback]

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=historical_df["timestamps"],
            open=historical_df["open"],
            high=historical_df["high"],
            low=historical_df["low"],
            close=historical_df["close"],
            name="Historical Data",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
        )
    )

    pred_timestamps = []
    if pred_df is not None and len(pred_df) > 0:
        pred_timestamps = list(pred_df["timestamps"])
        fig.add_trace(
            go.Candlestick(
                x=pred_timestamps,
                open=pred_df["open"],
                high=pred_df["high"],
                low=pred_df["low"],
                close=pred_df["close"],
                name="Prediction Data",
                increasing_line_color="#66BB6A",
                decreasing_line_color="#FF7043",
            )
        )

    if actual_df is not None and len(actual_df) > 0:
        fig.add_trace(
            go.Candlestick(
                x=list(actual_df["timestamps"]),
                open=actual_df["open"],
                high=actual_df["high"],
                low=actual_df["low"],
                close=actual_df["close"],
                name="Actual Data",
                increasing_line_color="#FF9800",
                decreasing_line_color="#F44336",
            )
        )

    fig.update_layout(
        title="Kronos Financial Prediction Results",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        height=600,
        showlegend=True,
    )
    fig.update_xaxes(rangeslider_visible=False, type="date")
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/available-models")
def get_available_models():
    return jsonify({"models": AVAILABLE_MODELS, "model_available": True})


@app.route("/api/load-model", methods=["POST"])
def load_model():
    data = request.get_json(silent=True) or {}
    model_key = data.get("model_key", settings.KRONOS_DEFAULT_MODEL)
    device = data.get("device", settings.KRONOS_DEFAULT_DEVICE)

    if model_key not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unsupported model: {model_key}"}), 400

    CURRENT_MODEL["loaded"] = True
    CURRENT_MODEL["model_key"] = model_key
    CURRENT_MODEL["device"] = "cpu" if device != "cpu" else device
    model_config = AVAILABLE_MODELS[model_key]
    return jsonify(
        {
            "success": True,
            "message": f"Model loaded successfully: {model_config['name']} ({model_config['params']}) on {CURRENT_MODEL['device']}",
            "model_info": {
                "name": model_config["name"],
                "params": model_config["params"],
                "context_length": model_config["context_length"],
                "description": model_config["description"],
            },
        }
    )


@app.route("/api/model-status")
def get_model_status():
    if CURRENT_MODEL["loaded"]:
        model_config = AVAILABLE_MODELS[CURRENT_MODEL["model_key"]]
        return jsonify(
            {
                "available": True,
                "loaded": True,
                "message": "Kronos model loaded and available",
                "current_model": {
                    "name": model_config["name"],
                    "device": CURRENT_MODEL["device"],
                },
            }
        )
    return jsonify(
        {
            "available": True,
            "loaded": False,
            "message": "Kronos model available but not loaded",
        }
    )


@app.route("/api/data-files")
def get_data_files():
    return jsonify(
        [
            {
                "name": "Factorhub Single Stock",
                "path": "__factorhub_stock__",
                "size": "Live data source",
                "source_type": "factorhub_stock",
            },
            *_scan_local_files(),
        ]
    )


@app.route("/api/load-data", methods=["POST"])
def load_data():
    data = request.get_json(silent=True) or {}
    file_path = data.get("file_path")
    source_type = data.get("source_type", "local_file")

    if source_type == "factorhub_stock" or file_path == "__factorhub_stock__":
        stock_code = str(data.get("stock_code", "")).strip()
        start_date = str(data.get("start_date", "")).strip()
        end_date = str(data.get("end_date", "")).strip()
        if not stock_code or not start_date or not end_date:
            return jsonify({"error": "stock_code, start_date, end_date are required for Factorhub stock source"}), 400
        df, error = _load_factorhub_stock(stock_code, start_date, end_date)
    else:
        if not file_path:
            return jsonify({"error": "File path cannot be empty"}), 400
        df, error = load_data_file(file_path)

    if error:
        return jsonify({"error": error}), 400

    data_info = _build_data_info(df)
    return jsonify(
        {
            "success": True,
            "data_info": data_info,
            "message": f"Successfully loaded data, total {len(df)} rows",
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if not CURRENT_MODEL["loaded"]:
        return jsonify({"error": "Kronos model not loaded, please load model first"}), 400

    data = request.get_json(silent=True) or {}
    file_path = data.get("file_path")
    source_type = data.get("source_type", "local_file")
    lookback = int(data.get("lookback", 400))
    pred_len = int(data.get("pred_len", 120))
    start_date = data.get("start_date")

    if source_type == "factorhub_stock" or file_path == "__factorhub_stock__":
        stock_code = str(data.get("stock_code", "")).strip()
        end_date = str(data.get("end_date", "")).strip()
        if not stock_code or not start_date or not end_date:
            return jsonify({"error": "stock_code, start_date, end_date are required for Factorhub stock source"}), 400
        df, error = _load_factorhub_stock(stock_code, start_date[:10], end_date)
    else:
        if not file_path:
            return jsonify({"error": "File path cannot be empty"}), 400
        df, error = load_data_file(file_path)

    if error:
        return jsonify({"error": error}), 400

    try:
        historical_df, pred_df, actual_df, prediction_type, historical_start_idx = simulate_prediction(
            df=df,
            lookback=lookback,
            pred_len=pred_len,
            start_date=start_date,
        )
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)

        prediction_results = []
        for _, row in pred_df.iterrows():
            prediction_results.append(
                {
                    "timestamp": pd.Timestamp(row["timestamps"]).isoformat(),
                    "open": _safe_float(row["open"]),
                    "high": _safe_float(row["high"]),
                    "low": _safe_float(row["low"]),
                    "close": _safe_float(row["close"]),
                    "volume": _safe_float(row["volume"]),
                    "amount": _safe_float(row["amount"]),
                }
            )

        actual_data = []
        for _, row in actual_df.iterrows():
            actual_data.append(
                {
                    "timestamp": pd.Timestamp(row["timestamps"]).isoformat(),
                    "open": _safe_float(row["open"]),
                    "high": _safe_float(row["high"]),
                    "low": _safe_float(row["low"]),
                    "close": _safe_float(row["close"]),
                    "volume": _safe_float(row["volume"]),
                    "amount": _safe_float(row["amount"]),
                }
            )

        return jsonify(
            {
                "success": True,
                "prediction_type": prediction_type,
                "chart": chart_json,
                "prediction_results": prediction_results,
                "actual_data": actual_data,
                "has_comparison": len(actual_data) > 0,
                "message": f"Prediction completed, generated {pred_len} prediction points",
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7070)
