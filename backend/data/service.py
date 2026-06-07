"""
数据服务模块。
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.core.settings import settings
from backend.data.providers import AkshareDataProvider, BaoStockDataProvider
from backend.data.registry import build_default_data_source_registry
from backend.services.cache_service import cache_service
from backend.services.data_preprocessing_service import data_preprocessing_service


class DataService:
    """数据服务类，负责股票数据获取和缓存。"""

    def __init__(self):
        self.cache_dir = settings.AKSHARE_CACHE_DIR
        self.universe_cache_dir = settings.DATA_UNIVERSE_CACHE_DIR
        self.benchmark_cache_dir = settings.DATA_BENCHMARK_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.universe_cache_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_service = cache_service
        self.preprocessing = data_preprocessing_service
        self.source_registry = build_default_data_source_registry()
        self.providers = {
            "akshare": AkshareDataProvider(),
            "baostock": BaoStockDataProvider(),
        }

    _BENCHMARK_CODES = {
        "hs300": {"baostock": "sh.000300", "name": "沪深300"},
        "zz500": {"baostock": "sh.000905", "name": "中证500"},
        "csi500": {"baostock": "sh.000905", "name": "中证500"},
        "csi1000": {"baostock": "sh.000852", "name": "中证1000"},
        "sz50": {"baostock": "sh.000016", "name": "上证50"},
    }

    def _get_cache_key(self, stock_code: str, start_date: str, end_date: str) -> str:
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        return hashlib.md5(cache_key.encode()).hexdigest()

    def _get_cache_path(self, stock_code: str, start_date: str, end_date: str) -> Path:
        cache_hash = self._get_cache_key(stock_code, start_date, end_date)
        return self.cache_dir / f"{cache_hash}.pkl"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        return self.cache_service.get(cache_key)

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str, ttl: Optional[int] = None) -> None:
        if ttl is None:
            ttl = settings.CACHE_DEFAULT_TTL
        self.cache_service.set(cache_key, data, ttl=ttl)

    def _to_akshare_symbol(self, stock_code: str) -> str:
        return self.providers["akshare"].to_symbol(stock_code)

    def _to_baostock_symbol(self, stock_code: str) -> str:
        return self.providers["baostock"].to_symbol(stock_code)

    def _fetch_from_akshare(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.providers["akshare"].fetch_stock_daily(stock_code, start_date, end_date)
        return self._standardize_columns(df)

    def _fetch_from_baostock(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.providers["baostock"].fetch_stock_daily(stock_code, start_date, end_date)
        df = df.rename(
            columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "amount": "amount",
                "pctChg": "pct_change",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        for col in ["open", "high", "low", "close", "volume", "amount", "pct_change"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_index()

    def _load_line_cache(self, cache_path: Path, min_count: int = 1) -> Optional[list[str]]:
        if not cache_path.exists():
            return None
        content = [line.strip() for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(content) < min_count:
            return None
        return content

    def _fetch_index_constituents_from_baostock(self, name: str, date: str) -> list[str]:
        return self.providers["baostock"].fetch_index_constituents(name, date)

    def _fetch_all_a_stocks_from_baostock(self, date: str) -> list[str]:
        return self.providers["baostock"].fetch_all_a_stocks(date)

    def get_supported_data_sources(self) -> dict:
        summary = self.source_registry.describe()
        summary["universes"] = ["hs300", "csi500", "zz500", "csi1000", "csi2000", "all_a"]
        return summary

    def _resolve_source_order(self, capability: str, preferred_source: Optional[str] = None) -> list[str]:
        candidates = [source.key for source in self.source_registry.get_sources_for_capability(capability)]
        if preferred_source:
            if preferred_source not in candidates:
                raise ValueError(f"数据源 {preferred_source} 不支持能力 {capability}")
            candidates = [preferred_source] + [item for item in candidates if item != preferred_source]
        return candidates

    def get_stock_universe(
        self,
        name: str,
        date: Optional[str] = None,
        preferred_source: Optional[str] = None,
    ) -> list[str]:
        normalized = name.strip().lower()
        date = date or datetime.now().strftime("%Y-%m-%d")
        cache_path = self.universe_cache_dir / f"{normalized}_{date[:7]}.txt"

        cached = self._load_line_cache(cache_path, min_count=10)
        if cached:
            return cached

        if normalized == "csi1000":
            hs300 = set(self.get_stock_universe("hs300", date))
            csi500 = set(self.get_stock_universe("csi500", date))
            all_a = self.get_stock_universe("all_a", date)
            codes = [code for code in all_a if code not in hs300 and code not in csi500][:1000]
        elif normalized == "csi2000":
            hs300 = set(self.get_stock_universe("hs300", date))
            csi500 = set(self.get_stock_universe("csi500", date))
            csi1000 = set(self.get_stock_universe("csi1000", date))
            all_a = self.get_stock_universe("all_a", date)
            codes = [code for code in all_a if code not in hs300 and code not in csi500 and code not in csi1000][:2000]
        else:
            codes = self._fetch_stock_universe_from_sources(normalized, date, preferred_source)

        if len(codes) >= 10:
            cache_path.write_text("\n".join(codes), encoding="utf-8")
        return codes

    def _fetch_stock_universe_from_sources(
        self,
        normalized_name: str,
        date: str,
        preferred_source: Optional[str] = None,
    ) -> list[str]:
        errors: list[str] = []
        for source_key in self._resolve_source_order("universe", preferred_source):
            try:
                if source_key != "baostock":
                    raise ValueError(f"暂未实现 {source_key} 的 universe provider")
                if normalized_name in ("hs300", "csi500", "zz500"):
                    return self._fetch_index_constituents_from_baostock(normalized_name, date)
                if normalized_name == "all_a":
                    return self._fetch_all_a_stocks_from_baostock(date)
            except Exception as exc:
                errors.append(f"{source_key}: {exc}")
        raise ValueError(f"暂不支持的股票池: {normalized_name}" if not errors else f"获取股票池 {normalized_name} 失败: {' | '.join(errors)}")

    def get_benchmark_returns(
        self,
        benchmark: str = "hs300",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        preferred_source: Optional[str] = None,
    ) -> pd.DataFrame:
        key = benchmark.strip().lower()
        info = self._BENCHMARK_CODES.get(key)
        if info is None:
            raise ValueError(f"未知 benchmark: {benchmark}")

        cache_path = self.benchmark_cache_dir / f"benchmark_{key}.pkl"
        start_date = start_date or (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        req_start = pd.Timestamp(start_date)
        req_end = pd.Timestamp(end_date)

        if cache_path.exists():
            try:
                cached_df = pd.read_pickle(cache_path)
                cached_df["trade_date"] = pd.to_datetime(cached_df["trade_date"])
                cached_df = cached_df.sort_values("trade_date")
                cache_min = cached_df["trade_date"].min()
                cache_max = cached_df["trade_date"].max()
                if cache_min <= req_start + pd.Timedelta(days=5) and cache_max >= req_end - pd.Timedelta(days=5):
                    return cached_df[
                        (cached_df["trade_date"] >= req_start) &
                        (cached_df["trade_date"] <= req_end)
                    ].reset_index(drop=True)
            except Exception:
                pass

        rows = self._fetch_benchmark_rows_from_sources(info["baostock"], start_date, end_date, preferred_source)
        if not rows:
            raise ValueError(f"未获取到 {benchmark} 基准数据")

        df = pd.DataFrame(rows, columns=["date", "close"])
        df["trade_date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.sort_values("trade_date")
        df["daily_return"] = df["close"].pct_change()
        df["benchmark"] = key
        df["daily_return"] = df["daily_return"].where(pd.notna(df["daily_return"]), None)
        result = df[["trade_date", "benchmark", "close", "daily_return"]].reset_index(drop=True)
        result.to_pickle(cache_path)
        return result

    def _fetch_benchmark_rows_from_sources(
        self,
        baostock_code: str,
        start_date: str,
        end_date: str,
        preferred_source: Optional[str] = None,
    ) -> list[list[str]]:
        errors: list[str] = []
        for source_key in self._resolve_source_order("benchmark", preferred_source):
            try:
                if source_key != "baostock":
                    raise ValueError(f"暂未实现 {source_key} 的 benchmark provider")
                rows = self.providers["baostock"].fetch_benchmark_rows(baostock_code, start_date, end_date)
                if rows:
                    return rows
            except Exception as exc:
                errors.append(f"{source_key}: {exc}")
        raise ValueError(f"获取 benchmark 数据失败: {' | '.join(errors)}")

    def get_stock_data(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        preferred_source: Optional[str] = None,
    ) -> pd.DataFrame:
        stock_code = self._normalize_stock_code(stock_code)

        if use_cache and settings.AKSHARE_CACHE_ENABLED:
            cache_key = self._get_cache_key(stock_code, start_date, end_date)
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        errors: list[str] = []
        df: Optional[pd.DataFrame] = None
        for source_key in self._resolve_source_order("stock_daily", preferred_source):
            try:
                if source_key == "akshare":
                    df = self._fetch_from_akshare(stock_code, start_date, end_date)
                elif source_key == "baostock":
                    df = self._fetch_from_baostock(stock_code, start_date, end_date)
                else:
                    raise ValueError(f"暂未实现 {source_key} 的 stock_daily provider")
                if df is not None and not df.empty:
                    break
            except Exception as exc:
                errors.append(f"{source_key}: {exc}")

        if df is None or df.empty:
            raise ValueError(f"获取股票 {stock_code} 数据失败: {' | '.join(errors)}")

        df = self._preprocess_data(df)
        if use_cache and settings.AKSHARE_CACHE_ENABLED:
            cache_key = self._get_cache_key(stock_code, start_date, end_date)
            self._save_to_cache(df, cache_key)

        return df

    def _normalize_stock_code(self, code: str) -> str:
        code = code.strip().upper()
        if not code.endswith((".SH", ".SZ")):
            if code.startswith("6"):
                return f"{code}.SH"
            if code.startswith(("0", "3")):
                return f"{code}.SZ"
        return code

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }
        df = df.rename(columns=column_mapping)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_index()

    def get_multiple_stocks_data(
        self,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        result = {}
        for code in stock_codes:
            try:
                df = self.get_stock_data(code, start_date, end_date, use_cache)
                result[code] = df
            except Exception as exc:
                print(f"Warning: 获取股票 {code} 数据失败: {exc}")
        return result

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if settings.DATA_FILL_MISSING:
            df = self.preprocessing.fill_missing_values(df, method=settings.DATA_FILL_METHOD)

        if settings.DATA_OUTLIER_DETECTION:
            df, _ = self.preprocessing.detect_and_handle_anomalies(
                df,
                price_columns=["open", "high", "low", "close"],
                n_sigma=settings.DATA_OUTLIER_N_SIGMA,
                handle_method=settings.DATA_OUTLIER_METHOD,
            )

        return df

    def get_cache_stats(self) -> dict:
        return self.cache_service.get_stats()

    def cleanup_cache(self) -> int:
        return self.cache_service.cleanup_expired()

    def clear_cache(self) -> int:
        return self.cache_service.clear_all()

    def incremental_update(self, stock_code: str, existing_df: pd.DataFrame, end_date: str) -> pd.DataFrame:
        last_date = existing_df.index.max()
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if start_date > end_date:
            return existing_df

        new_df = self.get_stock_data(
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
        )
        return self.preprocessing.incremental_update(existing_df=existing_df, new_df=new_df)


_data_service_instance: Optional[DataService] = None


def get_data_service() -> DataService:
    global _data_service_instance
    if _data_service_instance is None:
        _data_service_instance = DataService()
    return _data_service_instance


class _LazyDataServiceProxy:
    """兼容旧调用方式的惰性数据服务代理。"""

    def __getattr__(self, item):
        return getattr(get_data_service(), item)


data_service = _LazyDataServiceProxy()
