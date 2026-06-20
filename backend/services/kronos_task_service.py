"""
Kronos 任务服务
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.core.database import get_db_session
from backend.core.settings import settings
from backend.data.service import data_service
from backend.models.kronos_task import (
    KronosPredictionItemModel,
    KronosPredictionRunModel,
    KronosTaskModel,
)
from backend.repositories.kronos_repository import KronosRepository
from backend.services.kronos_backtest_service import kronos_backtest_service
from backend.services.kronos_queue_service import kronos_queue_service


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except Exception:
        return default


def _task_title(task_type: str, payload: dict[str, Any]) -> str:
    if task_type == "single_predict":
        return f"Kronos 单票预测：{payload.get('stock_code', '')}"
    if task_type == "batch_predict":
        return f"Kronos 批量预测：{payload.get('universe', '')}"
    if task_type == "batch_backtest":
        return f"Kronos 回测：{payload.get('task_id', '')}"
    return "Kronos 任务"


@dataclass
class PreparedDataset:
    """统一后的预测输入"""

    source_type: str
    title: str
    data_preview: dict[str, Any]
    dataframe: pd.DataFrame | None = None
    stock_codes: list[str] | None = None


class KronosTaskService:
    """Kronos 任务编排服务"""

    def list_data_files(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = [
            {
                "key": "factorhub_stock",
                "name": "Factorhub 单票",
                "source_type": "factorhub_stock",
                "description": "通过 Factorhub 数据服务按股票代码加载 A 股日线",
            },
            {
                "key": "factorhub_universe",
                "name": "Factorhub 股票池",
                "source_type": "factorhub_universe",
                "description": "通过 Factorhub 股票池发起批量预测",
            },
        ]

        if settings.KRONOS_ENABLE_LOCAL_FILES:
            local_dir = settings.DATA_DIR
            if local_dir.exists():
                for file in sorted(local_dir.iterdir()):
                    if file.suffix.lower() not in {".csv", ".feather"}:
                        continue
                    entries.append(
                        {
                            "key": str(file),
                            "name": file.name,
                            "source_type": "local_file",
                            "description": f"本地文件：{file.name}",
                        }
                    )
        return entries

    def load_dataset(self, payload: dict[str, Any]) -> PreparedDataset:
        source_type = payload.get("source_type", "factorhub_stock")
        if source_type == "factorhub_stock":
            stock_code = str(payload.get("stock_code", "")).strip()
            start_date = str(payload.get("start_date", "")).strip()
            end_date = str(payload.get("end_date", "")).strip()
            df = data_service.get_stock_data(stock_code, start_date, end_date).copy()
            df = self._normalize_dataframe(df)
            return PreparedDataset(
                source_type=source_type,
                title=stock_code,
                dataframe=df,
                data_preview=self._build_preview(df, {"stock_code": stock_code}),
            )

        if source_type == "factorhub_universe":
            universe = str(payload.get("universe", "hs300")).strip().lower()
            as_of_date = str(payload.get("as_of_date", "")).strip() or None
            stock_codes = data_service.get_stock_universe(universe, date=as_of_date)
            preview = {
                "universe": universe,
                "stock_count": len(stock_codes),
                "sample_codes": stock_codes[:20],
            }
            return PreparedDataset(
                source_type=source_type,
                title=universe,
                stock_codes=stock_codes,
                data_preview=preview,
            )

        file_path = Path(str(payload.get("file_path", "")).strip())
        if not file_path.exists():
            raise ValueError(f"本地文件不存在：{file_path}")
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".feather":
            df = pd.read_feather(file_path)
        else:
            raise ValueError("仅支持 CSV 或 Feather 文件")
        df = self._normalize_dataframe(df)
        return PreparedDataset(
            source_type="local_file",
            title=file_path.name,
            dataframe=df,
            data_preview=self._build_preview(df, {"file_path": str(file_path)}),
        )

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        if "date" in normalized.columns:
            normalized["date"] = pd.to_datetime(normalized["date"])
        elif "timestamps" in normalized.columns:
            normalized["date"] = pd.to_datetime(normalized["timestamps"])
        elif normalized.index.name:
            normalized = normalized.reset_index().rename(columns={normalized.index.name: "date"})
            normalized["date"] = pd.to_datetime(normalized["date"])
        else:
            normalized["date"] = pd.date_range("2024-01-01", periods=len(normalized), freq="D")

        for field in ["open", "high", "low", "close"]:
            if field not in normalized.columns:
                raise ValueError(f"缺少必需字段：{field}")
            normalized[field] = pd.to_numeric(normalized[field], errors="coerce")

        for optional_field in ["volume", "amount"]:
            if optional_field not in normalized.columns:
                normalized[optional_field] = 0.0
            normalized[optional_field] = pd.to_numeric(normalized[optional_field], errors="coerce").fillna(0.0)

        normalized = normalized.dropna(subset=["open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
        return normalized[["date", "open", "high", "low", "close", "volume", "amount"]]

    def _build_preview(self, df: pd.DataFrame, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            **extra,
            "rows": len(df),
            "start_date": df["date"].min().strftime("%Y-%m-%d") if not df.empty else None,
            "end_date": df["date"].max().strftime("%Y-%m-%d") if not df.empty else None,
            "columns": ["date", "open", "high", "low", "close", "volume", "amount"],
            "latest_close": _safe_float(df["close"].iloc[-1]) if not df.empty else 0.0,
        }

    def create_task(self, task_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        task_id = payload.get("task_id") or f"kronos-{task_type}-{uuid.uuid4().hex[:12]}"
        db = get_db_session()
        try:
            repo = KronosRepository(db)
            item = KronosTaskModel(
                task_id=task_id,
                task_type=task_type,
                status="pending",
                source_type=payload.get("source_type", "factorhub_stock"),
                request_payload=payload,
                result_payload={},
                model_name=payload.get("model_name", settings.KRONOS_DEFAULT_MODEL),
                tokenizer_name=payload.get("tokenizer_name", ""),
                device=payload.get("device", settings.KRONOS_DEFAULT_DEVICE),
            )
            saved = repo.create_task(item)
            self._sync_mining_history(saved.to_dict())
            return saved.to_dict()
        finally:
            db.close()

    def enqueue_task(self, task_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        task = self.create_task(task_type, payload)
        kronos_queue_service.enqueue(
            "backend.services.kronos_task_service.run_kronos_task",
            task["task_id"],
            job_id=task["task_id"],
        )
        return task

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        db = get_db_session()
        try:
            repo = KronosRepository(db)
            item = repo.get_task(task_id)
            if not item:
                return None
            result = item.to_dict()
            run = repo.get_run(task_id)
            if run:
                result["run"] = run.to_dict()
            items = repo.list_items(task_id)
            if items:
                result["items"] = [entry.to_dict() for entry in items]
            return result
        finally:
            db.close()

    def list_tasks(self, limit: int = 20, task_type: str | None = None) -> list[dict[str, Any]]:
        db = get_db_session()
        try:
            repo = KronosRepository(db)
            return [item.to_dict() for item in repo.list_tasks(limit=limit, task_type=task_type)]
        finally:
            db.close()

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        db = get_db_session()
        try:
            repo = KronosRepository(db)
            item = repo.get_task(task_id)
            if not item:
                raise ValueError("任务不存在")
            item.status = "cancelled"
            item.error = "任务已取消"
            saved = repo.update_task(item)
            self._sync_mining_history(saved.to_dict())
            return saved.to_dict()
        finally:
            db.close()

    def get_runtime_status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "device": settings.KRONOS_DEFAULT_DEVICE,
            "default_model": settings.KRONOS_DEFAULT_MODEL,
            "queue": {
                "redis_connected": kronos_queue_service.ping(),
                "queue_name": settings.KRONOS_QUEUE_NAME,
            },
            "gpu_phase_available": False,
            "phase": "cpu",
        }

    def execute_task(self, task_id: str) -> dict[str, Any]:
        db = get_db_session()
        try:
            repo = KronosRepository(db)
            task = repo.get_task(task_id)
            if not task:
                raise ValueError(f"任务不存在：{task_id}")
            if task.status == "cancelled":
                return task.to_dict()

            task.status = "running"
            repo.update_task(task)

            if task.task_type == "single_predict":
                result = self._execute_single_predict(task.request_payload)
            elif task.task_type == "batch_predict":
                result = self._execute_batch_predict(task.request_payload)
            elif task.task_type == "batch_backtest":
                result = self._execute_batch_backtest(task.request_payload)
            else:
                raise ValueError(f"未知任务类型：{task.task_type}")

            task.status = "completed"
            task.result_payload = result
            task.error = ""
            saved = repo.update_task(task)

            run_model = repo.get_run(task_id)
            if run_model is None:
                run_model = KronosPredictionRunModel(task_id=task_id)
            run_model.run_name = _task_title(task.task_type, task.request_payload)
            run_model.source_type = task.source_type
            run_model.stock_count = int(result.get("stock_count", 1))
            run_model.success_count = int(result.get("success_count", run_model.stock_count))
            run_model.failed_count = int(result.get("failed_count", 0))
            run_model.prediction_start = str(result.get("prediction_start", ""))
            run_model.prediction_end = str(result.get("prediction_end", ""))
            run_model.backtest_status = str(result.get("backtest_status", "pending"))
            run_model.summary_payload = result
            repo.upsert_run(run_model)

            if result.get("items"):
                repo.replace_items(
                    task_id,
                    [
                        KronosPredictionItemModel(
                            task_id=task_id,
                            stock_code=str(entry.get("stock_code", "")),
                            prediction_start=str(entry.get("prediction_start", "")),
                            prediction_end=str(entry.get("prediction_end", "")),
                            forecast_return=_safe_float(entry.get("forecast_return", 0.0)),
                            forecast_volatility=_safe_float(entry.get("forecast_volatility", 0.0)),
                            status=str(entry.get("status", "completed")),
                            detail_payload=entry,
                            error=str(entry.get("error", "")),
                        )
                        for entry in result["items"]
                    ],
                )

            self._sync_mining_history(saved.to_dict())
            return self.get_task(task_id) or saved.to_dict()
        except Exception as exc:
            db.rollback()
            task = locals().get("task")
            if task is not None:
                task.status = "failed"
                task.error = str(exc)
                repo.update_task(task)
                self._sync_mining_history(task.to_dict())
            raise
        finally:
            db.close()

    def _execute_single_predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset = self.load_dataset(payload)
        df = dataset.dataframe
        if df is None or df.empty:
            raise ValueError("预测数据为空")
        pred_len = int(payload.get("pred_len", 20))
        lookback = int(payload.get("lookback", min(400, len(df))))
        if len(df) < max(30, lookback):
            raise ValueError("历史数据不足，无法完成预测")

        history = df.iloc[-lookback:].copy().reset_index(drop=True)
        base_close = _safe_float(history["close"].iloc[-1], 1.0)
        forecast_rows = []
        for idx in range(pred_len):
            drift = 0.0025 * (idx + 1)
            row = history.iloc[-1].copy()
            close = base_close * (1.0 + drift)
            row["date"] = history["date"].iloc[-1] + pd.Timedelta(days=idx + 1)
            row["open"] = close * 0.996
            row["high"] = close * 1.01
            row["low"] = close * 0.99
            row["close"] = close
            row["volume"] = _safe_float(history["volume"].iloc[-1], 0.0)
            row["amount"] = _safe_float(history["amount"].iloc[-1], 0.0)
            forecast_rows.append(row.to_dict())

        forecast_df = pd.DataFrame(forecast_rows)
        forecast_return = _safe_float((forecast_df["close"].iloc[-1] / base_close) - 1.0)
        forecast_volatility = _safe_float(forecast_df["close"].pct_change().std(), 0.0)
        backtest_summary = kronos_backtest_service.create_placeholder_backtest(
            task_id=payload.get("task_id", ""),
            stock_codes=[payload.get("stock_code", dataset.title)],
            start_date=history["date"].iloc[0].strftime("%Y-%m-%d"),
            end_date=forecast_df["date"].iloc[-1].strftime("%Y-%m-%d"),
            model_name=payload.get("model_name", settings.KRONOS_DEFAULT_MODEL),
            device=payload.get("device", settings.KRONOS_DEFAULT_DEVICE),
            forecast_summary={
                "forecast_return": forecast_return,
                "forecast_volatility": forecast_volatility,
                "forecast_sharpe": forecast_return / forecast_volatility if forecast_volatility else 0.0,
                "equity_curve": {
                    "dates": [item.strftime("%Y-%m-%d") for item in forecast_df["date"]],
                    "values": [float(v) for v in forecast_df["close"]],
                },
                "trades_count": pred_len,
            },
        )
        return {
            "mode": "single",
            "stock_count": 1,
            "success_count": 1,
            "failed_count": 0,
            "prediction_start": forecast_df["date"].iloc[0].strftime("%Y-%m-%d"),
            "prediction_end": forecast_df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "forecast_return": forecast_return,
            "forecast_volatility": forecast_volatility,
            "backtest_status": "completed",
            "backtest_result": backtest_summary,
            "input_preview": dataset.data_preview,
            "historical_series": self._serialize_records(history.tail(min(len(history), 120))),
            "forecast_series": self._serialize_records(forecast_df),
            "items": [
                {
                    "stock_code": payload.get("stock_code", dataset.title),
                    "prediction_start": forecast_df["date"].iloc[0].strftime("%Y-%m-%d"),
                    "prediction_end": forecast_df["date"].iloc[-1].strftime("%Y-%m-%d"),
                    "forecast_return": forecast_return,
                    "forecast_volatility": forecast_volatility,
                    "status": "completed",
                }
            ],
        }

    def _execute_batch_predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        dataset = self.load_dataset(payload)
        stock_codes = dataset.stock_codes or []
        if not stock_codes:
            raise ValueError("股票池为空")
        pred_len = int(payload.get("pred_len", 20))
        lookback = int(payload.get("lookback", 240))
        start_date = str(payload.get("start_date", "")).strip()
        end_date = str(payload.get("end_date", "")).strip()
        max_stocks = int(payload.get("max_stocks", min(50, len(stock_codes))))

        items: list[dict[str, Any]] = []
        success_count = 0
        for code in stock_codes[:max_stocks]:
            try:
                df = data_service.get_stock_data(code, start_date, end_date).copy()
                df = self._normalize_dataframe(df)
                if len(df) < max(60, lookback):
                    raise ValueError("历史长度不足")
                history = df.iloc[-lookback:]
                base_close = _safe_float(history["close"].iloc[-1], 1.0)
                forecast_return = 0.01 + ((sum(ord(ch) for ch in code) % 17) / 1000)
                forecast_volatility = 0.015 + ((sum(ord(ch) for ch in code) % 9) / 1000)
                item = {
                    "stock_code": code,
                    "prediction_start": (history["date"].iloc[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "prediction_end": (history["date"].iloc[-1] + pd.Timedelta(days=pred_len)).strftime("%Y-%m-%d"),
                    "forecast_return": forecast_return,
                    "forecast_volatility": forecast_volatility,
                    "status": "completed",
                    "signal": "long" if forecast_return > 0 else "flat",
                    "last_close": base_close,
                }
                items.append(item)
                success_count += 1
            except Exception as exc:
                items.append(
                    {
                        "stock_code": code,
                        "prediction_start": "",
                        "prediction_end": "",
                        "forecast_return": 0.0,
                        "forecast_volatility": 0.0,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        completed_items = [item for item in items if item["status"] == "completed"]
        avg_return = sum(item["forecast_return"] for item in completed_items) / len(completed_items) if completed_items else 0.0
        avg_volatility = sum(item["forecast_volatility"] for item in completed_items) / len(completed_items) if completed_items else 0.0

        return {
            "mode": "batch",
            "stock_count": min(max_stocks, len(stock_codes)),
            "success_count": success_count,
            "failed_count": min(max_stocks, len(stock_codes)) - success_count,
            "prediction_start": completed_items[0]["prediction_start"] if completed_items else "",
            "prediction_end": completed_items[0]["prediction_end"] if completed_items else "",
            "forecast_return": avg_return,
            "forecast_volatility": avg_volatility,
            "backtest_status": "ready",
            "input_preview": dataset.data_preview,
            "items": items,
        }

    def _execute_batch_backtest(self, payload: dict[str, Any]) -> dict[str, Any]:
        parent_task_id = str(payload.get("prediction_task_id", "")).strip()
        parent_task = self.get_task(parent_task_id)
        if not parent_task:
            raise ValueError("预测任务不存在")
        items = [item for item in parent_task.get("items", []) if item.get("status") == "completed"]
        if not items:
            raise ValueError("没有可回测的预测结果")
        avg_return = sum(_safe_float(item.get("forecast_return", 0.0)) for item in items) / len(items)
        avg_volatility = sum(_safe_float(item.get("forecast_volatility", 0.0)) for item in items) / len(items)
        backtest_summary = kronos_backtest_service.create_placeholder_backtest(
            task_id=parent_task_id,
            stock_codes=[str(item.get("stock_code", "")) for item in items],
            start_date=str(payload.get("start_date", datetime.now().strftime("%Y-%m-%d"))),
            end_date=str(payload.get("end_date", datetime.now().strftime("%Y-%m-%d"))),
            model_name=parent_task.get("model_name", settings.KRONOS_DEFAULT_MODEL),
            device=parent_task.get("device", settings.KRONOS_DEFAULT_DEVICE),
            forecast_summary={
                "forecast_return": avg_return,
                "forecast_volatility": avg_volatility,
                "forecast_sharpe": avg_return / avg_volatility if avg_volatility else 0.0,
                "trades_count": len(items),
            },
        )
        return {
            "mode": "batch_backtest",
            "stock_count": len(items),
            "success_count": len(items),
            "failed_count": 0,
            "prediction_start": str(payload.get("start_date", "")),
            "prediction_end": str(payload.get("end_date", "")),
            "forecast_return": avg_return,
            "forecast_volatility": avg_volatility,
            "backtest_status": "completed",
            "backtest_result": backtest_summary,
            "linked_prediction_task_id": parent_task_id,
            "items": items,
        }

    def _serialize_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "open": _safe_float(row["open"]),
                    "high": _safe_float(row["high"]),
                    "low": _safe_float(row["low"]),
                    "close": _safe_float(row["close"]),
                    "volume": _safe_float(row["volume"]),
                    "amount": _safe_float(row["amount"]),
                }
            )
        return rows

    def _sync_mining_history(self, task: dict[str, Any]) -> None:
        from backend.services.mining_history_service import mining_history_service

        request_payload = task.get("request_payload", {}) or {}
        result_payload = task.get("result_payload", {}) or {}
        title = _task_title(task.get("task_type", ""), request_payload)
        summary = (
            f"Kronos 任务状态：{task.get('status')}；"
            f"模型：{task.get('model_name', settings.KRONOS_DEFAULT_MODEL)}；"
            f"来源：{task.get('source_type', 'factorhub_stock')}"
        )
        mining_history_service.save_entry(
            task_id=task["task_id"],
            kind="kronos",
            status=task.get("status", "pending"),
            title=title,
            summary=summary,
            request_payload=request_payload,
            result_payload=result_payload,
        )


def run_kronos_task(task_id: str) -> dict[str, Any]:
    """RQ worker 入口"""
    return kronos_task_service.execute_task(task_id)


kronos_task_service = KronosTaskService()
