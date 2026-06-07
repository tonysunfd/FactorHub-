"""
标准数据源 provider 实现。

`DataService` 只负责调度、缓存、预处理；
各 provider 负责具体的数据抓取。
"""
from __future__ import annotations

from datetime import datetime, timedelta
import threading
import time
from typing import Protocol

import pandas as pd


class StockDailyProvider(Protocol):
    """股票日线 provider 协议。"""

    def fetch_stock_daily(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """抓取单只股票日线数据。"""


class UniverseProvider(Protocol):
    """股票池 provider 协议。"""

    def fetch_index_constituents(self, name: str, date: str) -> list[str]:
        """抓取指数成分股。"""

    def fetch_all_a_stocks(self, date: str) -> list[str]:
        """抓取全市场 A 股代码。"""


class BenchmarkProvider(Protocol):
    """benchmark provider 协议。"""

    def fetch_benchmark_rows(self, code: str, start_date: str, end_date: str) -> list[list[str]]:
        """抓取 benchmark 原始行。"""


class FundamentalProvider(Protocol):
    """财务数据 provider 协议。"""

    def fetch_fundamental_quarters(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        apis: set[str],
        api_func_map: dict[str, str],
        api_fields: dict[str, list[str]],
    ) -> pd.DataFrame | None:
        """抓取季度财务数据。"""


class DividendProvider(Protocol):
    """分红数据 provider 协议。"""

    def fetch_dividend_events(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        """抓取分红事件。"""


class IndustryProvider(Protocol):
    """行业分类 provider 协议。"""

    def fetch_industry_data(self, stock_codes: list[str]) -> pd.DataFrame | None:
        """抓取行业分类。"""


class AkshareDataProvider:
    """AKShare provider。"""

    def to_symbol(self, stock_code: str) -> str:
        if stock_code.endswith(".SH"):
            return "sh" + stock_code.replace(".SH", "")
        if stock_code.endswith(".SZ"):
            return "sz" + stock_code.replace(".SZ", "")
        if stock_code.startswith("6"):
            return "sh" + stock_code
        if stock_code.startswith(("0", "3")):
            return "sz" + stock_code
        return stock_code

    def fetch_stock_daily(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak

        symbol = self.to_symbol(stock_code)
        return ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",
        )


class BaoStockDataProvider:
    """BaoStock provider。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def to_symbol(self, stock_code: str) -> str:
        if stock_code.endswith(".SH"):
            return "sh." + stock_code.replace(".SH", "")
        if stock_code.endswith(".SZ"):
            return "sz." + stock_code.replace(".SZ", "")
        if stock_code.startswith("6"):
            return "sh." + stock_code
        if stock_code.startswith(("0", "3")):
            return "sz." + stock_code
        return stock_code.lower()

    def _login(self) -> None:
        import baostock as bs

        for attempt in range(3):
            try:
                login_result = bs.login()
                if getattr(login_result, "error_code", "0") == "0":
                    return
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                raise ValueError(getattr(login_result, "error_msg", "BaoStock 登录失败"))
            except Exception:
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                raise

    def _logout(self) -> None:
        try:
            import baostock as bs

            bs.logout()
        except Exception:
            pass

    def fetch_stock_daily(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import baostock as bs

        symbol = self.to_symbol(stock_code)
        with self._lock:
            self._login()
            try:
                rs = bs.query_history_k_data_plus(
                    symbol,
                    "date,code,open,high,low,close,volume,amount,pctChg",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="2",
                )
                if getattr(rs, "error_code", "0") != "0":
                    raise ValueError(getattr(rs, "error_msg", "BaoStock 查询失败"))
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
            finally:
                self._logout()

        if not rows:
            raise ValueError("BaoStock 未返回数据")
        return pd.DataFrame(rows, columns=rs.fields)

    def fetch_index_constituents(self, name: str, date: str) -> list[str]:
        import baostock as bs

        with self._lock:
            self._login()
            try:
                if name == "hs300":
                    rs = bs.query_hs300_stocks(date)
                else:
                    rs = bs.query_zz500_stocks(date)
                codes: list[str] = []
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    if len(row) > 1 and row[1]:
                        codes.append(row[1])
                return codes
            finally:
                self._logout()

    def fetch_all_a_stocks(self, date: str) -> list[str]:
        import baostock as bs

        with self._lock:
            self._login()
            try:
                base_date = datetime.strptime(date, "%Y-%m-%d")
                for offset in range(0, 30):
                    try_date = (base_date - timedelta(days=offset)).strftime("%Y-%m-%d")
                    rs = bs.query_all_stock(day=try_date)
                    codes: list[str] = []
                    while rs.error_code == "0" and rs.next():
                        row = rs.get_row_data()
                        if not row:
                            continue
                        code = row[0]
                        if not (code.startswith("sh.") or code.startswith("sz.")):
                            continue
                        if code.startswith("sh.000") or code.startswith("bj."):
                            continue
                        codes.append(code)
                    if len(codes) > 100:
                        return codes
                return []
            finally:
                self._logout()

    def fetch_benchmark_rows(self, code: str, start_date: str, end_date: str) -> list[list[str]]:
        import baostock as bs

        with self._lock:
            self._login()
            try:
                rs = bs.query_history_k_data_plus(
                    code,
                    "date,close",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="2",
                )
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                return rows
            finally:
                self._logout()

    def fetch_industry_data(self, stock_codes: list[str]) -> pd.DataFrame | None:
        import baostock as bs

        rows: list[dict] = []
        with self._lock:
            self._login()
            try:
                for code in stock_codes:
                    rs = bs.query_stock_industry(code=code)
                    while rs.error_code == "0" and rs.next():
                        row = rs.get_row_data()
                        if len(row) >= 4:
                            rows.append(
                                {
                                    "stock_code": row[1],
                                    "industry_code": row[2] if len(row) > 2 else "",
                                    "industry": row[3] if len(row) > 3 else "其他",
                                }
                            )
                        break
            finally:
                self._logout()

        if not rows:
            return None
        return pd.DataFrame(rows).drop_duplicates(subset=["stock_code"], keep="last")

    def fetch_fundamental_quarters(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        apis: set[str],
        api_func_map: dict[str, str],
        api_fields: dict[str, list[str]],
    ) -> pd.DataFrame | None:
        if not apis:
            return None

        import baostock as bs

        start = datetime.strptime(start_date[:10], "%Y-%m-%d")
        end = datetime.strptime(end_date[:10], "%Y-%m-%d")
        quarters = [(year, quarter) for year in range(start.year - 1, end.year + 1) for quarter in range(1, 5)]

        results: list[pd.DataFrame] = []
        with self._lock:
            self._login()
            try:
                for api_name in apis:
                    api_frames: list[pd.DataFrame] = []
                    func = getattr(bs, api_func_map[api_name])
                    for year, quarter in quarters:
                        rs = func(code=stock_code, year=year, quarter=quarter)
                        if getattr(rs, "error_code", "0") != "0":
                            continue
                        rows = []
                        while rs.next():
                            rows.append(rs.get_row_data())
                        if rows:
                            api_frames.append(pd.DataFrame(rows, columns=rs.fields))
                    if api_frames:
                        frame = pd.concat(api_frames, ignore_index=True)
                        keep_cols = [col for col in api_fields[api_name] if col in frame.columns]
                        results.append(frame[keep_cols].copy())
            finally:
                self._logout()

        if not results:
            return None

        merged = results[0]
        for frame in results[1:]:
            merge_on = ["code", "pubDate", "statDate"]
            extras = [col for col in frame.columns if col not in merged.columns]
            if extras:
                merged = merged.merge(frame[merge_on + extras], on=merge_on, how="outer")
        return merged

    def fetch_dividend_events(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
        import baostock as bs

        start_year = datetime.strptime(start_date[:10], "%Y-%m-%d").year - 1
        end_year = datetime.strptime(end_date[:10], "%Y-%m-%d").year
        rows: list[dict] = []

        with self._lock:
            self._login()
            try:
                for year in range(start_year, end_year + 1):
                    rs = bs.query_dividend_data(code=stock_code, year=str(year), yearType="report")
                    if getattr(rs, "error_code", "0") != "0":
                        continue
                    while rs.next():
                        row = rs.get_row_data()
                        if len(row) < 10 or not row[6] or not row[9]:
                            continue
                        try:
                            cash_per_share = float(row[9])
                        except (TypeError, ValueError):
                            continue
                        if cash_per_share <= 0:
                            continue
                        rows.append(
                            {
                                "stock_code": stock_code,
                                "ex_date": pd.Timestamp(row[6]),
                                "cash_per_share": cash_per_share,
                            }
                        )
            finally:
                self._logout()

        if not rows:
            return None

        dividend_df = pd.DataFrame(rows)
        dividend_df = dividend_df.drop_duplicates(subset=["stock_code", "ex_date", "cash_per_share"])
        return dividend_df.sort_values("cash_per_share", ascending=False).drop_duplicates(
            subset=["stock_code", "ex_date"], keep="first"
        )
