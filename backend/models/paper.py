from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class PaperStrategyModel(Base):
    __tablename__ = "paper_strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    backtest_id: Mapped[int] = mapped_column(Integer, ForeignKey("backtest_results.id"), nullable=False)
    strategy_config: Mapped[dict] = mapped_column(JSON, nullable=False)

    initial_capital: Mapped[float] = mapped_column(Float, default=1_000_000.0)
    current_value: Mapped[float] = mapped_column(Float, default=1_000_000.0)
    commission_rate: Mapped[float] = mapped_column(Float, default=0.0003)
    stamp_tax_rate: Mapped[float] = mapped_column(Float, default=0.001)
    slippage_rate: Mapped[float] = mapped_column(Float, default=0.0)

    status: Mapped[str] = mapped_column(String(20), default="active")
    last_rebalance_date: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    next_rebalance_date: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class PaperSnapshotModel(Base):
    __tablename__ = "paper_snapshots"
    __table_args__ = (
        UniqueConstraint("strategy_id", "date", name="uq_paper_snapshots_strategy_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper_strategies.id"), nullable=False)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, default=0.0)
    market_value: Mapped[float] = mapped_column(Float, default=0.0)
    daily_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    positions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class PaperOrderModel(Base):
    __tablename__ = "paper_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper_strategies.id"), nullable=False)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    stock_code: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    shares: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
