"""
通用行情增强服务。

负责变量识别、缓存、日频对齐和派生字段计算；
具体数据抓取通过标准 provider 完成。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.core.settings import settings
from backend.data.providers import BaoStockDataProvider

logger = logging.getLogger(__name__)


FUNDAMENTAL_VARIABLES: dict[str, tuple[str, str]] = {
    "roe": ("profit", "roeAvg"),
    "np_margin": ("profit", "npMargin"),
    "gp_margin": ("profit", "gpMargin"),
    "net_profit": ("profit", "netProfit"),
    "eps_ttm": ("profit", "epsTTM"),
    "revenue": ("profit", "MBRevenue"),
    "total_share": ("profit", "totalShare"),
    "float_share": ("profit", "liqaShare"),
    "yoy_ni": ("growth", "YOYNI"),
    "yoy_equity": ("growth", "YOYEquity"),
    "yoy_asset": ("growth", "YOYAsset"),
    "yoy_pni": ("growth", "YOYPNI"),
    "current_ratio": ("balance", "currentRatio"),
    "debt_ratio": ("balance", "liabilityToAsset"),
    "equity_multiplier": ("balance", "assetToEquity"),
    "asset_turnover": ("operation", "AssetTurnRatio"),
    "inv_turnover": ("operation", "INVTurnRatio"),
    "dupont_roe": ("dupont", "dupontROE"),
    "dupont_asset_turn": ("dupont", "dupontAssetTurn"),
    "cfo_to_np": ("cash_flow", "CFOToNP"),
}

DERIVED_VARIABLES: dict[str, list[str]] = {
    "pe": ["net_profit", "total_share"],
    "pb": ["net_profit", "total_share", "roe"],
    "ps": ["revenue", "total_share"],
    "roa": ["roe", "equity_multiplier"],
    "bps": ["net_profit", "total_share", "roe"],
    "nav": ["net_profit", "roe"],
    "market_cap": ["total_share"],
    "float_market_cap": ["float_share"],
}

SPECIAL_VARIABLES = {"dividend_yield", "industry", "industry_code"}
ALL_SUPPORTED_VARIABLES = frozenset(FUNDAMENTAL_VARIABLES) | frozenset(DERIVED_VARIABLES) | SPECIAL_VARIABLES

_API_FUNC_MAP = {
    "profit": "query_profit_data",
    "growth": "query_growth_data",
    "balance": "query_balance_data",
    "operation": "query_operation_data",
    "dupont": "query_dupont_data",
    "cash_flow": "query_cash_flow_data",
}

_API_FIELDS: dict[str, list[str]] = {
    "profit": ["code", "pubDate", "statDate", "roeAvg", "npMargin", "gpMargin", "netProfit", "epsTTM", "MBRevenue", "totalShare", "liqaShare"],
    "growth": ["code", "pubDate", "statDate", "YOYNI", "YOYEquity", "YOYAsset", "YOYPNI"],
    "balance": ["code", "pubDate", "statDate", "currentRatio", "liabilityToAsset", "assetToEquity"],
    "operation": ["code", "pubDate", "statDate", "AssetTurnRatio", "INVTurnRatio"],
    "dupont": ["code", "pubDate", "statDate", "dupontROE", "dupontAssetTurn"],
    "cash_flow": ["code", "pubDate", "statDate", "CFOToNP"],
}

_SOURCE_TO_USER: dict[str, str] = {field: name for name, (_, field) in FUNDAMENTAL_VARIABLES.items()}


class MarketDataEnrichmentService:
    """通用行情增强服务。"""

    def __init__(self, provider: BaoStockDataProvider | None = None) -> None:
        self.provider = provider or BaoStockDataProvider()
        self.fundamental_cache_dir = settings.DATA_FUNDAMENTAL_CACHE_DIR
        self.dividend_cache_dir = settings.DATA_DIVIDEND_CACHE_DIR
        self.industry_cache_dir = settings.DATA_INDUSTRY_CACHE_DIR
        for path in (self.fundamental_cache_dir, self.dividend_cache_dir, self.industry_cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    def detect_variables(self, expressions: list[str]) -> set[str]:
        tokens: set[str] = set()
        for expression in expressions:
            tokens.update(re.findall(r"\b[a-z_]+\b", expression.lower()))
        return tokens & ALL_SUPPORTED_VARIABLES

    def enrich_daily_data(
        self,
        market_df: pd.DataFrame,
        stock_code: str,
        start_date: str,
        end_date: str,
        needed_vars: set[str],
    ) -> pd.DataFrame:
        if market_df is None or market_df.empty or not needed_vars:
            return market_df

        enriched = market_df.copy()
        if "trade_date" not in enriched.columns:
            enriched = enriched.reset_index().rename(columns={"date": "trade_date"})
        enriched["trade_date"] = pd.to_datetime(enriched["trade_date"])
        enriched["stock_code"] = stock_code

        fundamental_vars = {name for name in needed_vars if name in FUNDAMENTAL_VARIABLES or name in DERIVED_VARIABLES}
        if fundamental_vars:
            fundamentals = self._fetch_fundamental_daily(stock_code, start_date, end_date, enriched, fundamental_vars)
            if fundamentals is not None and not fundamentals.empty:
                for column in fundamentals.columns:
                    if column in {"trade_date", "stock_code"}:
                        continue
                    enriched[column] = fundamentals[column].values

        if "dividend_yield" in needed_vars:
            dividend_series = self._fetch_dividend_yield(stock_code, start_date, end_date, enriched)
            if dividend_series is not None:
                enriched["dividend_yield"] = dividend_series.values

        if {"industry", "industry_code"} & needed_vars:
            industry_df = self.get_industry_data([stock_code])
            if industry_df is not None and not industry_df.empty:
                row = industry_df[industry_df["stock_code"] == stock_code]
                if not row.empty:
                    enriched["industry"] = row.iloc[0]["industry"]
                    enriched["industry_code"] = row.iloc[0]["industry_code"]

        if "market_cap" in needed_vars and "market_cap" not in enriched.columns:
            if "total_share" in enriched.columns:
                enriched["market_cap"] = pd.to_numeric(enriched["close"], errors="coerce") * pd.to_numeric(enriched["total_share"], errors="coerce")
            elif "volume" in enriched.columns:
                enriched["market_cap"] = pd.to_numeric(enriched["close"], errors="coerce") * pd.to_numeric(enriched["volume"], errors="coerce")

        if "float_market_cap" in needed_vars and "float_market_cap" not in enriched.columns and "float_share" in enriched.columns:
            enriched["float_market_cap"] = pd.to_numeric(enriched["close"], errors="coerce") * pd.to_numeric(enriched["float_share"], errors="coerce")

        enriched = enriched.set_index("trade_date")
        enriched.index.name = "date"
        return enriched

    def get_industry_data(self, stock_codes: list[str]) -> Optional[pd.DataFrame]:
        month_key = datetime.now().strftime("%Y-%m")
        cache_path = self.industry_cache_dir / f"industry_{month_key}.pkl"
        cached = self._read_pickle(cache_path)
        if cached is not None and not cached.empty:
            cached_codes = set(cached["stock_code"].astype(str))
            if set(stock_codes).issubset(cached_codes):
                return cached

        try:
            fresh = self.provider.fetch_industry_data(stock_codes)
        except Exception as exc:
            logger.warning("市场行业分类获取失败：%s", exc)
            return cached

        if fresh is None or fresh.empty:
            return cached

        if cached is not None and not cached.empty:
            fresh = pd.concat([cached, fresh], ignore_index=True).drop_duplicates(subset=["stock_code"], keep="last")
        fresh.to_pickle(cache_path)
        return fresh

    def _fetch_fundamental_daily(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        market_df: pd.DataFrame,
        needed_vars: set[str],
    ) -> Optional[pd.DataFrame]:
        raw_vars = self._expand_raw_vars(needed_vars)
        if not raw_vars:
            return None

        quarter_df = self._load_fundamental_cache(stock_code)
        missing_apis = self._required_apis(raw_vars)
        if quarter_df is None or quarter_df.empty or not raw_vars.issubset(set(quarter_df.columns)):
            fetched = self._fetch_fundamental_quarters(stock_code, start_date, end_date, missing_apis)
            if fetched is not None and not fetched.empty:
                if quarter_df is not None and not quarter_df.empty:
                    quarter_df = pd.concat([quarter_df, fetched], ignore_index=True)
                    quarter_df = quarter_df.sort_values("pub_date").drop_duplicates(subset=["stock_code", "stat_date"], keep="last")
                else:
                    quarter_df = fetched
                self._save_fundamental_cache(stock_code, quarter_df)

        if quarter_df is None or quarter_df.empty:
            return None

        keep_cols = ["stock_code", "pub_date"] + [name for name in raw_vars if name in quarter_df.columns]
        aligned_source = quarter_df[keep_cols].dropna(subset=["pub_date"]).copy()
        if aligned_source.empty:
            return None

        market_sorted = market_df.sort_values("trade_date").copy()
        merged = pd.merge_asof(
            market_sorted,
            aligned_source.sort_values("pub_date").drop(columns=["stock_code"]),
            left_on="trade_date",
            right_on="pub_date",
            direction="backward",
        )

        self._apply_derived_fields(merged, needed_vars)
        merged = merged.drop(columns=["pub_date"], errors="ignore")
        numeric_cols = [name for name in needed_vars if name in merged.columns]
        return merged[["trade_date", "stock_code"] + numeric_cols]

    def _fetch_dividend_yield(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        market_df: pd.DataFrame,
    ) -> Optional[pd.Series]:
        dividend_df = self._load_dividend_cache(stock_code)
        if dividend_df is None or dividend_df.empty:
            dividend_df = self._fetch_dividend_events(stock_code, start_date, end_date)
            if dividend_df is not None and not dividend_df.empty:
                self._save_dividend_cache(stock_code, dividend_df)

        if dividend_df is None or dividend_df.empty:
            return None

        market_sorted = market_df.sort_values("trade_date").copy()
        div_dates = dividend_df["ex_date"].to_numpy()
        div_cash = dividend_df["cash_per_share"].to_numpy()
        values: list[float] = []
        for trade_date, close_price in zip(market_sorted["trade_date"], market_sorted["close"]):
            cutoff = pd.Timestamp(trade_date) - pd.Timedelta(days=365)
            mask = (div_dates >= cutoff.to_numpy()) & (div_dates <= pd.Timestamp(trade_date).to_numpy())
            ttm_div = float(div_cash[mask].sum()) if mask.any() else np.nan
            if pd.isna(ttm_div) or not close_price:
                values.append(np.nan)
            else:
                values.append(ttm_div / close_price)
        return pd.Series(values, index=market_sorted.index, dtype=float)

    def _fetch_fundamental_quarters(self, stock_code: str, start_date: str, end_date: str, apis: set[str]) -> Optional[pd.DataFrame]:
        merged = self.provider.fetch_fundamental_quarters(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            apis=apis,
            api_func_map=_API_FUNC_MAP,
            api_fields=_API_FIELDS,
        )
        if merged is None or merged.empty:
            return None

        rename_map = {"code": "stock_code", "pubDate": "pub_date", "statDate": "stat_date"}
        for field_name, user_name in _SOURCE_TO_USER.items():
            if field_name in merged.columns:
                rename_map[field_name] = user_name
        merged = merged.rename(columns=rename_map)
        for column in merged.columns:
            if column in {"stock_code", "pub_date", "stat_date"}:
                continue
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        merged["pub_date"] = pd.to_datetime(merged["pub_date"], errors="coerce")
        merged["stat_date"] = pd.to_datetime(merged["stat_date"], errors="coerce")
        merged = merged.dropna(subset=["pub_date"])
        return merged.sort_values("pub_date").drop_duplicates(subset=["stock_code", "stat_date"], keep="last")

    def _fetch_dividend_events(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        return self.provider.fetch_dividend_events(stock_code, start_date, end_date)

    def _load_fundamental_cache(self, stock_code: str) -> Optional[pd.DataFrame]:
        return self._read_pickle(self.fundamental_cache_dir / f"{stock_code.replace('.', '_')}.pkl")

    def _save_fundamental_cache(self, stock_code: str, df: pd.DataFrame) -> None:
        df.to_pickle(self.fundamental_cache_dir / f"{stock_code.replace('.', '_')}.pkl")

    def _load_dividend_cache(self, stock_code: str) -> Optional[pd.DataFrame]:
        return self._read_pickle(self.dividend_cache_dir / f"{stock_code.replace('.', '_')}.pkl")

    def _save_dividend_cache(self, stock_code: str, df: pd.DataFrame) -> None:
        df.to_pickle(self.dividend_cache_dir / f"{stock_code.replace('.', '_')}.pkl")

    def _expand_raw_vars(self, needed_vars: set[str]) -> set[str]:
        raw_vars: set[str] = set()
        for name in needed_vars:
            if name in DERIVED_VARIABLES:
                raw_vars.update(DERIVED_VARIABLES[name])
            elif name in FUNDAMENTAL_VARIABLES:
                raw_vars.add(name)
        return raw_vars

    def _required_apis(self, raw_vars: set[str]) -> set[str]:
        return {FUNDAMENTAL_VARIABLES[name][0] for name in raw_vars if name in FUNDAMENTAL_VARIABLES}

    def _apply_derived_fields(self, merged: pd.DataFrame, needed_vars: set[str]) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            if "pe" in needed_vars and {"close", "total_share", "net_profit"} <= set(merged.columns):
                merged["pe"] = np.where(merged["net_profit"].notna() & (merged["net_profit"] != 0), merged["close"] * merged["total_share"] / merged["net_profit"], np.nan)
            if "pb" in needed_vars and {"close", "total_share", "net_profit", "roe"} <= set(merged.columns):
                book_value = np.where(merged["roe"].notna() & (merged["roe"] != 0), merged["net_profit"] / merged["roe"], np.nan)
                merged["pb"] = np.where(pd.notna(book_value) & (book_value != 0), merged["close"] * merged["total_share"] / book_value, np.nan)
            if "ps" in needed_vars and {"close", "total_share", "revenue"} <= set(merged.columns):
                merged["ps"] = np.where(merged["revenue"].notna() & (merged["revenue"] != 0), merged["close"] * merged["total_share"] / merged["revenue"], np.nan)
            if "roa" in needed_vars and {"roe", "equity_multiplier"} <= set(merged.columns):
                merged["roa"] = np.where(merged["equity_multiplier"].notna() & (merged["equity_multiplier"] != 0), merged["roe"] / merged["equity_multiplier"], np.nan)
            if "bps" in needed_vars and {"net_profit", "total_share", "roe"} <= set(merged.columns):
                book_value = np.where(merged["roe"].notna() & (merged["roe"] != 0), merged["net_profit"] / merged["roe"], np.nan)
                merged["bps"] = np.where(merged["total_share"].notna() & (merged["total_share"] != 0), book_value / merged["total_share"], np.nan)
            if "nav" in needed_vars and {"net_profit", "roe"} <= set(merged.columns):
                merged["nav"] = np.where(merged["roe"].notna() & (merged["roe"] != 0), merged["net_profit"] / merged["roe"], np.nan)
            if "market_cap" in needed_vars and {"close", "total_share"} <= set(merged.columns):
                merged["market_cap"] = merged["close"] * merged["total_share"]
            if "float_market_cap" in needed_vars and {"close", "float_share"} <= set(merged.columns):
                merged["float_market_cap"] = merged["close"] * merged["float_share"]

    def _read_pickle(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_pickle(path)
            for column in ("pub_date", "stat_date", "ex_date"):
                if column in df.columns:
                    df[column] = pd.to_datetime(df[column], errors="coerce")
            return df
        except Exception as exc:
            logger.warning("读取行情增强缓存失败：%s", exc)
            return None


market_data_enrichment_service = MarketDataEnrichmentService()
