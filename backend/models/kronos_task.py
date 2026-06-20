"""
Kronos 预测任务相关数据模型
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class KronosTaskModel(Base):
    """Kronos 异步任务主表"""

    __tablename__ = "kronos_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="factorhub_stock")
    request_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    result_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, default="kronos-base")
    tokenizer_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    device: Mapped[str] = mapped_column(String(32), nullable=False, default="cpu")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "source_type": self.source_type,
            "request_payload": self.request_payload or {},
            "result_payload": self.result_payload or {},
            "error": self.error,
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "device": self.device,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class KronosPredictionRunModel(Base):
    """Kronos 预测批次摘要表"""

    __tablename__ = "kronos_prediction_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    run_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="factorhub_stock")
    stock_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prediction_start: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    prediction_end: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    backtest_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    summary_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "run_name": self.run_name,
            "source_type": self.source_type,
            "stock_count": self.stock_count,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "prediction_start": self.prediction_start,
            "prediction_end": self.prediction_end,
            "backtest_status": self.backtest_status,
            "summary_payload": self.summary_payload or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class KronosPredictionItemModel(Base):
    """Kronos 批量预测逐标的结果表"""

    __tablename__ = "kronos_prediction_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    stock_code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    prediction_start: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    prediction_end: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    forecast_return: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    forecast_volatility: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    detail_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "stock_code": self.stock_code,
            "prediction_start": self.prediction_start,
            "prediction_end": self.prediction_end,
            "forecast_return": self.forecast_return,
            "forecast_volatility": self.forecast_volatility,
            "status": self.status,
            "detail_payload": self.detail_payload or {},
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
