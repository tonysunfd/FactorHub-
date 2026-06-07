"""
回测结果数据模型
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base


class BacktestResultModel(Base):
    """回测结果模型"""
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False, comment="策略名称")
    factor_combination: Mapped[str] = mapped_column(String(500), nullable=False, comment="因子组合（JSON字符串）")
    start_date: Mapped[str] = mapped_column(String(20), nullable=False, comment="开始日期")
    end_date: Mapped[str] = mapped_column(String(20), nullable=False, comment="结束日期")
    initial_capital: Mapped[float] = mapped_column(Float, default=1000000, comment="初始资金")
    final_capital: Mapped[float] = mapped_column(Float, nullable=True, comment="最终资金")
    total_return: Mapped[float] = mapped_column(Float, nullable=True, comment="累计收益率")
    annual_return: Mapped[float] = mapped_column(Float, nullable=True, comment="年化收益率")
    volatility: Mapped[float] = mapped_column(Float, nullable=True, comment="年化波动率")
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=True, comment="夏普比率")
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=True, comment="最大回撤")
    calmar_ratio: Mapped[float] = mapped_column(Float, nullable=True, comment="卡玛比率")
    win_rate: Mapped[float] = mapped_column(Float, nullable=True, comment="胜率")
    sortino_ratio: Mapped[float] = mapped_column(Float, nullable=True, comment="索提诺比率")
    equity_curve: Mapped[dict] = mapped_column(JSON, nullable=True, comment="净值曲线数据")
    quantile_returns: Mapped[dict] = mapped_column(JSON, nullable=True, comment="各分层收益数据")
    trades_count: Mapped[int] = mapped_column(Integer, default=0, comment="交易次数")
    benchmark_return: Mapped[float] = mapped_column(Float, nullable=True, comment="基准收益率")
    excess_return: Mapped[float] = mapped_column(Float, nullable=True, comment="超额收益率")
    strategy_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="结构化策略定义")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="创建时间")

    def __repr__(self):
        return f"<BacktestResult(id={self.id}, strategy={self.strategy_name}, return={self.total_return:.2%})>"


class TradeRecordModel(Base):
    """交易记录模型"""
    __tablename__ = "trade_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_id: Mapped[int] = mapped_column(Integer, ForeignKey("backtest_results.id"), nullable=False, comment="回测结果ID")
    stock_code: Mapped[str] = mapped_column(String(20), nullable=False, comment="股票代码")
    trade_date: Mapped[str] = mapped_column(String(20), nullable=False, comment="交易日期")
    action: Mapped[str] = mapped_column(String(10), nullable=False, comment="交易动作（buy/sell）")
    price: Mapped[float] = mapped_column(Float, nullable=False, comment="成交价格")
    shares: Mapped[int] = mapped_column(Integer, nullable=False, comment="成交数量")
    amount: Mapped[float] = mapped_column(Float, nullable=False, comment="成交金额")
    commission: Mapped[float] = mapped_column(Float, default=0, comment="手续费")
    factor_value: Mapped[float] = mapped_column(Float, nullable=True, comment="因子值")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="创建时间")

    def __repr__(self):
        return f"<TradeRecord(stock={self.stock_code}, date={self.trade_date}, action={self.action})>"
