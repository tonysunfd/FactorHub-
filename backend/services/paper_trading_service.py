from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from backend.models.paper import PaperOrderModel, PaperSnapshotModel
from backend.repositories.paper_repository import PaperRepository
from backend.data.service import data_service
from backend.services.factor_service import factor_service

MIN_COMMISSION = 5.0


class PaperTradingService:
    """FactorHub-native paper trading service with tighter QuantGPT alignment."""

    def __init__(self, db):
        self.db = db
        self.repo = PaperRepository(db)

    def settle_all_active_strategies(self):
        strategies = [s for s in self.repo.list_strategies() if s.status == "active"]
        results = []
        for strategy in strategies:
            results.append(self.settle_strategy(strategy.id))
        return results

    def settle_strategy(self, strategy_id: int, force: bool = False):
        strategy = self.repo.get_strategy(strategy_id)
        if not strategy:
            return None

        config: dict[str, Any] = strategy.strategy_config or {}
        strategy_class = config.get("strategy_class", "factor_rotation")
        if strategy_class != "factor_rotation":
            raise ValueError(f"暂不支持的策略类型: {strategy_class}")

        today = datetime.now().strftime("%Y-%m-%d")
        latest_snapshot = self.repo.get_latest_snapshot(strategy.id)
        if latest_snapshot and latest_snapshot.date == today and not force:
            return strategy

        should_rebalance = (
            latest_snapshot is None
            or not strategy.last_rebalance_date
            or (strategy.next_rebalance_date and today >= strategy.next_rebalance_date)
            or force
        )

        if should_rebalance:
            self._settle_rebalance_day(strategy, today, latest_snapshot, config)
        else:
            self._settle_hold_day(strategy, today, latest_snapshot)

        self.db.commit()
        self.db.refresh(strategy)
        return strategy

    def _settle_hold_day(self, strategy, today: str, latest_snapshot: Optional[PaperSnapshotModel]):
        if latest_snapshot is None:
            self._upsert_snapshot(
                strategy_id=strategy.id,
                date=today,
                portfolio_value=strategy.initial_capital,
                cash=strategy.initial_capital,
                market_value=0.0,
                daily_return=0.0,
                positions={},
            )
            strategy.current_value = strategy.initial_capital
            strategy.updated_at = datetime.now()
            return

        prev_positions = latest_snapshot.positions or {}
        cash = float(latest_snapshot.cash or 0.0)
        prev_nav = float(latest_snapshot.portfolio_value or 0.0)

        if not prev_positions:
            self._upsert_snapshot(
                strategy_id=strategy.id,
                date=today,
                portfolio_value=cash,
                cash=cash,
                market_value=0.0,
                daily_return=0.0,
                positions={},
            )
            strategy.current_value = cash
            strategy.updated_at = datetime.now()
            return

        close_prices = self._fetch_latest_prices(list(prev_positions.keys()), today, today, field="close")
        market_value = 0.0
        for code, pos in prev_positions.items():
            shares = float((pos or {}).get("shares", 0))
            fallback_price = float((pos or {}).get("entry_price", 0.0))
            mark_price = close_prices.get(code, fallback_price)
            market_value += shares * mark_price

        nav = cash + market_value
        daily_return = (nav / prev_nav - 1) if prev_nav else 0.0
        self._upsert_snapshot(
            strategy_id=strategy.id,
            date=today,
            portfolio_value=nav,
            cash=cash,
            market_value=market_value,
            daily_return=daily_return,
            positions=prev_positions,
        )
        strategy.current_value = nav
        strategy.updated_at = datetime.now()

    def _settle_rebalance_day(self, strategy, today: str, latest_snapshot: Optional[PaperSnapshotModel], config: dict[str, Any]):
        stock_codes = config.get("stock_codes") or []
        factor_names = config.get("factor_names") or []
        primary_factor = config.get("factor_name")
        strategy_type = config.get("strategy_type", "single_factor")
        direction = config.get("direction", "long")
        weight_method = config.get("weight_method", "equal_weight")
        shares_per_trade = int(config.get("shares_per_trade", 100) or 100)
        holding_period = max(1, int(config.get("holding_period", 5) or 5))
        n_groups = max(2, int(config.get("n_groups", config.get("n_quantiles", 5)) or 5))
        end_date = today
        lookback_start = (pd.Timestamp(today) - pd.Timedelta(days=380)).strftime("%Y-%m-%d")

        if not stock_codes:
            raise ValueError("模拟盘策略缺少 stock_codes，无法结算")

        if strategy_type == "single_factor":
            if not primary_factor:
                raise ValueError("单因子策略缺少 factor_name")
            factor_names_to_use = [primary_factor]
        else:
            factor_names_to_use = factor_names or ([primary_factor] if primary_factor else [])
            if not factor_names_to_use:
                raise ValueError("多因子策略缺少 factor_names")

        all_factor_data = factor_service.calculate_factors_for_stocks(
            stock_codes=stock_codes,
            factor_names=factor_names_to_use,
            start_date=lookback_start,
            end_date=end_date,
        )
        if not all_factor_data:
            raise ValueError("未获取到可用于模拟盘结算的因子数据")

        latest_rows: list[dict[str, Any]] = []
        for code, df in all_factor_data.items():
            if df is None or len(df) == 0:
                continue
            working = df.copy()
            if not isinstance(working.index, pd.DatetimeIndex):
                if "date" in working.columns:
                    working = working.set_index("date")
                working.index = pd.to_datetime(working.index)
            working = working.sort_index()
            working = working.loc[working.index <= pd.Timestamp(today)]
            if len(working) == 0 or "close" not in working.columns:
                continue

            latest_row = working.iloc[-1]
            open_price = self._pick_price(latest_row, "open")
            close_price = self._pick_price(latest_row, "close")
            if close_price is None:
                continue

            if strategy_type == "single_factor":
                score = working[primary_factor].iloc[-1] if primary_factor in working.columns else None
            else:
                available = [f for f in factor_names_to_use if f in working.columns]
                if not available:
                    continue
                z_last = []
                for col in available:
                    series = pd.to_numeric(working[col], errors="coerce")
                    mean = series.mean()
                    std = series.std()
                    val = series.iloc[-1]
                    if pd.isna(val):
                        continue
                    z_last.append(0.0 if pd.isna(std) or std == 0 else float((val - mean) / std))
                if not z_last:
                    continue
                score = sum(z_last) / len(z_last) if weight_method == "equal_weight" else sum(z_last) / len(z_last)

            if pd.isna(score):
                continue

            latest_rows.append({
                "stock_code": code,
                "score": float(score),
                "open": float(open_price) if open_price is not None else None,
                "close": float(close_price),
            })

        if not latest_rows:
            raise ValueError("没有可用于调仓的最新股票评分")

        latest_df = pd.DataFrame(latest_rows).dropna(subset=["close"])
        latest_df = latest_df.sort_values("score", ascending=(direction != "long"))
        n_per_group = max(1, len(latest_df) // n_groups)
        selected = latest_df.head(n_per_group).copy()

        prev_positions = (latest_snapshot.positions or {}) if latest_snapshot else {}
        prev_cash = float(latest_snapshot.cash) if latest_snapshot and latest_snapshot.cash is not None else float(strategy.initial_capital)
        prev_nav = float(latest_snapshot.portfolio_value) if latest_snapshot and latest_snapshot.portfolio_value is not None else float(strategy.initial_capital)

        cash = prev_cash
        unsold: dict[str, dict[str, Any]] = {}

        for code, pos in prev_positions.items():
            shares = float((pos or {}).get("shares", 0))
            price_row = latest_df[latest_df["stock_code"] == code]
            open_price = None if price_row.empty else price_row["open"].iloc[0]
            if shares <= 0 or open_price is None or pd.isna(open_price) or float(open_price) <= 0:
                unsold[code] = pos
                continue
            price = float(open_price)
            amount = shares * price
            commission = max(amount * float(strategy.commission_rate or 0.0), MIN_COMMISSION)
            stamp_tax = amount * float(strategy.stamp_tax_rate or 0.0)
            slippage = amount * float(strategy.slippage_rate or 0.0)
            cash += amount - commission - stamp_tax - slippage
            self.db.add(PaperOrderModel(
                strategy_id=strategy.id,
                date=today,
                stock_code=code,
                direction="sell",
                shares=shares,
                price=price,
                amount=amount,
                commission=commission + stamp_tax,
                slippage=slippage,
            ))

        new_positions: dict[str, dict[str, Any]] = {}
        buyable = selected.dropna(subset=["open"])
        if not buyable.empty:
            per_stock_cash = cash / len(buyable)
            for _, row in buyable.iterrows():
                code = str(row["stock_code"])
                price = float(row["open"])
                if price <= 0:
                    continue
                lot_size = max(100, shares_per_trade)
                max_shares = int(per_stock_cash / (price * (1 + float(strategy.commission_rate or 0.0) + float(strategy.slippage_rate or 0.0))) / lot_size) * lot_size
                if max_shares < lot_size:
                    continue
                amount = max_shares * price
                commission = max(amount * float(strategy.commission_rate or 0.0), MIN_COMMISSION)
                slippage = amount * float(strategy.slippage_rate or 0.0)
                total_cost = amount + commission + slippage
                if total_cost > cash:
                    continue
                cash -= total_cost
                new_positions[code] = {
                    "shares": max_shares,
                    "entry_price": price,
                    "entry_date": today,
                    "score": float(row["score"]),
                }
                self.db.add(PaperOrderModel(
                    strategy_id=strategy.id,
                    date=today,
                    stock_code=code,
                    direction="buy",
                    shares=max_shares,
                    price=price,
                    amount=amount,
                    commission=commission,
                    slippage=slippage,
                ))

        for code, pos in unsold.items():
            if code not in new_positions:
                new_positions[code] = pos

        market_value = 0.0
        for code, pos in new_positions.items():
            shares = float((pos or {}).get("shares", 0))
            price_row = latest_df[latest_df["stock_code"] == code]
            close_price = None if price_row.empty else price_row["close"].iloc[0]
            mark_price = float(close_price) if close_price is not None and not pd.isna(close_price) and float(close_price) > 0 else float((pos or {}).get("entry_price", 0.0))
            market_value += shares * mark_price

        nav = cash + market_value
        daily_return = (nav / prev_nav - 1) if prev_nav else 0.0

        next_rebalance_date = (pd.Timestamp(today) + pd.tseries.offsets.BDay(holding_period)).strftime("%Y-%m-%d")
        strategy.last_rebalance_date = today
        strategy.next_rebalance_date = next_rebalance_date
        strategy.current_value = nav
        strategy.updated_at = datetime.now()

        self._upsert_snapshot(
            strategy_id=strategy.id,
            date=today,
            portfolio_value=nav,
            cash=cash,
            market_value=market_value,
            daily_return=daily_return,
            positions=new_positions,
        )

    def _fetch_latest_prices(self, stock_codes: list[str], start_date: str, end_date: str, field: str = "close") -> dict[str, float]:
        prices: dict[str, float] = {}
        for code in stock_codes:
            try:
                data = data_service.get_stock_data(code, start_date, end_date)
            except Exception:
                continue
            if data is None or len(data) == 0:
                continue
            latest = data.iloc[-1]
            price = self._pick_price(latest, field)
            if price is not None:
                prices[code] = float(price)
        return prices

    def _pick_price(self, row: Any, field: str) -> Optional[float]:
        value = None
        if isinstance(row, pd.Series):
            value = row.get(field)
        elif isinstance(row, dict):
            value = row.get(field)
        if value is None or pd.isna(value):
            return None
        return float(value)

    def _upsert_snapshot(
        self,
        strategy_id: int,
        date: str,
        portfolio_value: float,
        cash: float,
        market_value: float,
        daily_return: float,
        positions: dict[str, Any],
    ) -> PaperSnapshotModel:
        snapshot = self.repo.get_snapshot_by_date(strategy_id, date)
        if snapshot is None:
            snapshot = PaperSnapshotModel(
                strategy_id=strategy_id,
                date=date,
                portfolio_value=portfolio_value,
                cash=cash,
                market_value=market_value,
                daily_return=daily_return,
                positions=positions,
            )
            self.db.add(snapshot)
        else:
            snapshot.portfolio_value = portfolio_value
            snapshot.cash = cash
            snapshot.market_value = market_value
            snapshot.daily_return = daily_return
            snapshot.positions = positions
        return snapshot
