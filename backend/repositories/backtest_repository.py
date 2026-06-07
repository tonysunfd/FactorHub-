"""
回测结果数据访问层
"""
from typing import Optional, List, Dict, Any
from sqlalchemy import select, delete, desc
from datetime import datetime

from backend.models.backtest import BacktestResultModel, TradeRecordModel
from backend.core.database import get_db_session


class BacktestRepository:
    """回测结果数据访问类"""

    def __init__(self):
        self.db = get_db_session()

    def save_result(self, result_data: Dict[str, Any]) -> BacktestResultModel:
        """
        保存回测结果

        Args:
            result_data: 回测结果字典，包含所有回测指标

        Returns:
            BacktestResultModel: 保存的回测结果对象
        """
        # 序列化 JSON 字段
        equity_curve = result_data.get("equity_curve")
        if equity_curve is not None and hasattr(equity_curve, "to_dict"):
            result_data["equity_curve"] = equity_curve.to_dict()

        quantile_returns = result_data.get("quantile_returns", {})
        serialized_quantile = {}
        for key, value in quantile_returns.items():
            if hasattr(value, "to_dict"):
                serialized_quantile[key] = value.to_dict()
            elif isinstance(value, dict):
                serialized_quantile[key] = {
                    k: v.to_dict() if hasattr(v, "to_dict") else v
                    for k, v in value.items()
                }
            else:
                serialized_quantile[key] = value
        result_data["quantile_returns"] = serialized_quantile

        # 创建回测结果对象
        backtest_result = BacktestResultModel(
            strategy_name=result_data.get("strategy_name", "未命名策略"),
            factor_combination=result_data.get("factor_combination", ""),
            start_date=result_data.get("start_date", ""),
            end_date=result_data.get("end_date", ""),
            initial_capital=result_data.get("initial_capital", 1000000),
            final_capital=result_data.get("final_capital"),
            total_return=result_data.get("total_return"),
            annual_return=result_data.get("annual_return"),
            volatility=result_data.get("volatility"),
            sharpe_ratio=result_data.get("sharpe_ratio"),
            max_drawdown=result_data.get("max_drawdown"),
            calmar_ratio=result_data.get("calmar_ratio"),
            win_rate=result_data.get("win_rate"),
            sortino_ratio=result_data.get("sortino_ratio"),
            equity_curve=result_data.get("equity_curve"),
            quantile_returns=result_data.get("quantile_returns"),
            trades_count=result_data.get("trades_count", 0),
            benchmark_return=result_data.get("benchmark_return"),
            excess_return=result_data.get("excess_return"),
            strategy_config=result_data.get("strategy_config"),
        )

        self.db.add(backtest_result)
        self.db.commit()
        self.db.refresh(backtest_result)

        return backtest_result

    def get_history(self, limit: int = 20, offset: int = 0) -> List[BacktestResultModel]:
        """
        获取历史回测记录

        Args:
            limit: 返回记录数量
            offset: 偏移量

        Returns:
            List[BacktestResultModel]: 回测结果列表
        """
        query = (
            select(BacktestResultModel)
            .order_by(desc(BacktestResultModel.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(query).all())

    def get_by_id(self, result_id: int) -> Optional[BacktestResultModel]:
        """
        根据ID获取回测结果

        Args:
            result_id: 回测结果ID

        Returns:
            Optional[BacktestResultModel]: 回测结果对象或None
        """
        return self.db.scalar(
            select(BacktestResultModel).where(BacktestResultModel.id == result_id)
        )

    def delete_by_id(self, result_id: int) -> bool:
        """
        删除回测结果及其关联的交易记录

        Args:
            result_id: 回测结果ID

        Returns:
            bool: 是否删除成功
        """
        try:
            # 先删除关联的交易记录
            self.db.execute(
                delete(TradeRecordModel).where(TradeRecordModel.backtest_id == result_id)
            )
            # 删除回测结果
            self.db.execute(
                delete(BacktestResultModel).where(BacktestResultModel.id == result_id)
            )
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            return False

    def save_trade_records(
        self, backtest_id: int, trades: List[Dict[str, Any]]
    ) -> int:
        """
        批量保存交易记录

        Args:
            backtest_id: 回测结果ID
            trades: 交易记录列表

        Returns:
            int: 保存的记录数
        """
        count = 0
        for trade in trades:
            trade_record = TradeRecordModel(
                backtest_id=backtest_id,
                stock_code=trade.get("stock_code", ""),
                trade_date=trade.get("trade_date", ""),
                action=trade.get("action", "buy"),
                price=trade.get("price", 0.0),
                shares=trade.get("shares", 0),
                amount=trade.get("amount", 0.0),
                commission=trade.get("commission", 0.0),
                factor_value=trade.get("factor_value"),
            )
            self.db.add(trade_record)
            count += 1

        self.db.commit()
        return count

    def get_trade_records(self, backtest_id: int) -> List[TradeRecordModel]:
        """
        获取指定回测的交易记录

        Args:
            backtest_id: 回测结果ID

        Returns:
            List[TradeRecordModel]: 交易记录列表
        """
        query = select(TradeRecordModel).where(
            TradeRecordModel.backtest_id == backtest_id
        ).order_by(TradeRecordModel.trade_date)
        return list(self.db.scalars(query).all())

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取回测统计信息

        Returns:
            Dict: 统计信息字典
        """
        total_count = self.db.scalar(select(BacktestResultModel.id).count()) or 0

        # 计算平均收益
        avg_return = self.db.scalar(
            select(BacktestResultModel.total_return).where(
                BacktestResultModel.total_return.isnot(None)
            )
        )
        if avg_return is not None:
            avg_return = float(avg_return)

        # 计算平均夏普比率
        avg_sharpe = self.db.scalar(
            select(BacktestResultModel.sharpe_ratio).where(
                BacktestResultModel.sharpe_ratio.isnot(None)
            )
        )
        if avg_sharpe is not None:
            avg_sharpe = float(avg_sharpe)

        return {
            "total_count": total_count,
            "avg_return": avg_return,
            "avg_sharpe": avg_sharpe,
        }

    def close(self):
        """关闭数据库连接"""
        self.db.close()
