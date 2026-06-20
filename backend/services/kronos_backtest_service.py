"""
Kronos 预测结果与回测联动服务
"""
from __future__ import annotations

from typing import Any

from backend.repositories.backtest_repository import BacktestRepository


class KronosBacktestService:
    """将 Kronos 预测摘要写入现有回测结果体系"""

    def create_placeholder_backtest(
        self,
        *,
        task_id: str,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        model_name: str,
        device: str,
        forecast_summary: dict[str, Any],
    ) -> dict[str, Any]:
        repo = BacktestRepository()
        try:
            saved = repo.save_result(
                {
                    "strategy_name": f"Kronos {'单票' if len(stock_codes) == 1 else '批量'}预测回测",
                    "factor_combination": "kronos_forecast_signal",
                    "start_date": start_date,
                    "end_date": end_date,
                    "initial_capital": 1_000_000,
                    "final_capital": 1_000_000,
                    "total_return": float(forecast_summary.get("forecast_return", 0.0)),
                    "annual_return": float(forecast_summary.get("forecast_return", 0.0)),
                    "volatility": float(forecast_summary.get("forecast_volatility", 0.0)),
                    "sharpe_ratio": float(forecast_summary.get("forecast_sharpe", 0.0)),
                    "max_drawdown": float(forecast_summary.get("max_drawdown", 0.0)),
                    "trades_count": int(forecast_summary.get("trades_count", 0)),
                    "equity_curve": forecast_summary.get("equity_curve", {}),
                    "quantile_returns": {},
                    "strategy_config": {
                        "source": "kronos",
                        "kronos_task_id": task_id,
                        "stock_codes": stock_codes,
                        "model_name": model_name,
                        "device": device,
                    },
                }
            )
            return {
                "backtest_id": saved.id,
                "strategy_name": saved.strategy_name,
                "total_return": saved.total_return,
                "annual_return": saved.annual_return,
                "volatility": saved.volatility,
                "sharpe_ratio": saved.sharpe_ratio,
            }
        finally:
            repo.db.close()


kronos_backtest_service = KronosBacktestService()
