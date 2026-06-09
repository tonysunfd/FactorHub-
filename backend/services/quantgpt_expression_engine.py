from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from backend.data.enrichment import ALL_SUPPORTED_VARIABLES, market_data_enrichment_service
from backend.services.expression_schema import EngineExecutionResult
from backend.services.research_tools.expression_adapter import ExpressionAdapter

logger = logging.getLogger(__name__)

PRICE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_change",
    "market_cap",
    "float_market_cap",
    "turnover_rate",
    "vwap",
    "returns",
    "cap",
    "day",
    "weekday",
    "month",
    "industry",
    "industry_code",
    "stock_code",
    "trade_date",
    "date",
}
ALLOWED_COLUMNS = PRICE_COLUMNS | set(ALL_SUPPORTED_VARIABLES)

_ALIAS_NORMALIZE = {
    "delta": "ts_delta",
    "delay": "ts_shift",
    "stddev": "ts_std",
    "covariance": "ts_cov",
    "correlation": "ts_corr",
    "ts_decay_linear": "decay_linear",
    "ts_product": "product",
    "ts_delay": "ts_shift",
    "ts_covariance": "ts_cov",
    "ts_arg_max": "ts_argmax",
    "ts_arg_min": "ts_argmin",
}

_OP_PATTERN = re.compile(r"([a-z_][a-z0-9_]*)\s*\(")
_FIELD_PATTERN = re.compile(r"\b([a-z_][a-z0-9_]*)\b")


def normalize_expression(expression: str) -> str:
    expr = re.sub(r"\s+", "", expression.lower())
    for alias, canonical in _ALIAS_NORMALIZE.items():
        expr = re.sub(rf"\b{re.escape(alias)}\b", canonical, expr)
    return expr


def extract_components(expression: str) -> dict[str, Any]:
    expr_lower = expression.lower()
    operators = set(_OP_PATTERN.findall(expr_lower))
    all_words = set(_FIELD_PATTERN.findall(expr_lower))
    all_ops = operators | {"if", "else", "and", "or", "true", "false"}
    fields = {word for word in all_words - all_ops if not word.replace(".", "").isdigit()}
    return {"operators": sorted(operators), "fields": sorted(fields)}


class ExpressionParser:
    MAX_WINDOW = 500
    MAX_DEPTH = 100
    MAX_EXPRESSION_LENGTH = 1000

    _OPERATOR_ALIASES = {
        "delta": "ts_delta",
        "delay": "ts_shift",
        "covariance": "ts_cov",
        "correlation": "ts_corr",
        "av_diff": "ts_av_diff",
        "stddev": "ts_std",
        "ts_decay_linear": "decay_linear",
        "ts_product": "product",
        "ts_std_dev": "ts_std",
        "ts_delay": "ts_shift",
        "ts_covariance": "ts_cov",
        "ts_arg_max": "ts_argmax",
        "ts_arg_min": "ts_argmin",
    }

    _SPECIAL_VARS = {
        "vwap": lambda df: df["vwap"] if "vwap" in df.columns else (df["amount"] / df["volume"].replace(0, np.nan) if "amount" in df.columns else df["close"]),
        "returns": lambda df: df.groupby("stock_code")["close"].pct_change() if "stock_code" in df.columns else df["close"].pct_change(),
        "cap": lambda df: df["market_cap"] if "market_cap" in df.columns else df["close"] * df.get("total_share", 1),
        "day": lambda df: pd.Series(df["trade_date"].dt.day, index=df.index, dtype=float),
        "weekday": lambda df: pd.Series(df["trade_date"].dt.weekday, index=df.index, dtype=float),
        "month": lambda df: pd.Series(df["trade_date"].dt.month, index=df.index, dtype=float),
    }

    _CROSS_SECTIONAL_OPS = {"rank", "zscore"}

    _UNARY_OPS = {
        "log": lambda s: np.log(s.clip(lower=1e-10)),
        "abs": lambda s: s.abs(),
        "sign": lambda s: np.sign(s),
        "scale": lambda s: (s - s.min()) / (s.max() - s.min() + 1e-10),
        "tanh": lambda s: np.tanh(s),
        "sigmoid": lambda s: 1.0 / (1.0 + np.exp(-s.clip(-500, 500))),
        "exp": lambda s: np.exp(s.clip(upper=500)),
        "sqrt": lambda s: np.sqrt(s.clip(lower=0)),
    }

    @staticmethod
    def _calc_rsi(s: pd.Series, w: int) -> pd.Series:
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(w, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(w, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_macd(s: pd.Series, w: int) -> pd.Series:
        fast = max(2, w // 2)
        signal = max(2, w // 4)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=w, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    @staticmethod
    def _calc_atr(df: pd.DataFrame, w: int) -> pd.Series:
        high = df.get("high", df["close"])
        low = df.get("low", df["close"])
        close_prev = df["close"].shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - close_prev).abs(),
                (low - close_prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(w, min_periods=1).mean()

    _TS_OPS = {
        "ts_mean": lambda s, w: s.rolling(w, min_periods=1).mean(),
        "ts_std": lambda s, w: s.rolling(w, min_periods=1).std(),
        "ts_max": lambda s, w: s.rolling(w, min_periods=1).max(),
        "ts_min": lambda s, w: s.rolling(w, min_periods=1).min(),
        "ts_sum": lambda s, w: s.rolling(w, min_periods=1).sum(),
        "ts_shift": lambda s, w: s.shift(w),
        "ts_delta": lambda s, w: s - s.shift(w),
        "ts_rank": lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False),
        "ts_argmax": lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: x.argmax(), raw=True),
        "ts_argmin": lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: x.argmin(), raw=True),
        "decay_linear": lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)) if len(x) > 0 else np.nan, raw=True),
        "product": lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: np.prod(x), raw=True),
        "ts_av_diff": lambda s, w: s - s.rolling(w, min_periods=1).mean(),
        "ts_zscore": lambda s, w: (s - s.rolling(w, min_periods=1).mean()) / (s.rolling(w, min_periods=1).std() + 1e-10),
        "ema": lambda s, w: s.ewm(span=w, adjust=False).mean(),
        "sma": lambda s, w: s.rolling(w, min_periods=1).mean(),
        "rsi": lambda s, w: ExpressionParser._calc_rsi(s, w),
        "macd": lambda s, w: ExpressionParser._calc_macd(s, w),
        "obv": lambda s, w: s.rolling(w, min_periods=1).sum(),
        "wma": lambda s, w: s.rolling(w, min_periods=1).apply(lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)) if len(x) > 0 else np.nan, raw=True),
    }

    _TS_DUAL_OPS = {
        "ts_corr": lambda s1, s2, w: s1.rolling(w, min_periods=1).corr(s2),
        "ts_cov": lambda s1, s2, w: s1.rolling(w, min_periods=1).cov(s2),
    }

    _BINARY_OPS = {
        "power": lambda s, exp: s ** exp,
        "pow": lambda s, exp: s ** exp,
        "sign_power": lambda s, exp: np.sign(s) * (np.abs(s) ** exp),
        "max": lambda a, b: np.maximum(a, b),
        "min": lambda a, b: np.minimum(a, b),
    }

    def parse(self, expression: str, _depth: int = 0) -> Callable[[pd.DataFrame], pd.Series]:
        if _depth > self.MAX_DEPTH:
            raise ValueError(f"Expression nesting too deep (max {self.MAX_DEPTH})")
        expression = expression.strip()
        if len(expression) > self.MAX_EXPRESSION_LENGTH:
            raise ValueError(f"Expression too long (max {self.MAX_EXPRESSION_LENGTH} chars)")
        self._depth = _depth
        if _depth == 0:
            expression = self._convert_ternary_operators(expression)

        func_match = self._match_function_call(expression)
        if func_match is not None:
            func_name, args_str, remainder = func_match
            if not remainder:
                return self._build_function(func_name, args_str)
        return self._build_arithmetic(expression)

    @staticmethod
    def _match_function_call(expression: str) -> tuple[str, str, str] | None:
        matched = re.match(r"^(\w+)\(", expression)
        if not matched:
            return None
        func_name = matched.group(1).lower()
        start = matched.end() - 1
        depth = 0
        for index in range(start, len(expression)):
            if expression[index] == "(":
                depth += 1
            elif expression[index] == ")":
                depth -= 1
                if depth == 0:
                    return func_name, expression[start + 1:index], expression[index + 1 :].strip()
        return None

    def _sub_parse(self, expr: str) -> Callable[[pd.DataFrame], pd.Series]:
        return self.parse(expr, self._depth + 1)

    def _validate_window(self, window: int, func_name: str) -> int:
        if window < 1:
            raise ValueError(f"{func_name}: window must be >= 1, got {window}")
        if window > self.MAX_WINDOW:
            raise ValueError(f"{func_name}: window too large (max {self.MAX_WINDOW}), got {window}")
        return window

    def _parse_window_value(self, raw_value: str, func_name: str) -> int:
        value = raw_value.strip()
        if "=" in value:
            _, value = value.split("=", 1)
        return self._validate_window(int(value.strip()), func_name)

    @staticmethod
    def _apply_ts_op_per_stock(df: pd.DataFrame, inner_fn: Callable[[pd.DataFrame], pd.Series], op: Callable[[pd.Series, int], pd.Series], window: int) -> pd.Series:
        series = inner_fn(df)
        if "stock_code" in df.columns:
            return series.groupby(df["stock_code"]).transform(lambda x: op(x, window))
        return op(series, window)

    def _build_function(self, func_name: str, args_str: str) -> Callable[[pd.DataFrame], pd.Series]:
        func_name = self._OPERATOR_ALIASES.get(func_name, func_name)

        if func_name in self._CROSS_SECTIONAL_OPS:
            inner = self._sub_parse(args_str)
            if func_name == "rank":
                def _cs_rank(df: pd.DataFrame, _inner: Callable[[pd.DataFrame], pd.Series] = inner) -> pd.Series:
                    series = _inner(df)
                    if "trade_date" in df.columns:
                        return series.groupby(df["trade_date"]).rank(pct=True)
                    return series.rank(pct=True)
                return _cs_rank

            def _cs_zscore(df: pd.DataFrame, _inner: Callable[[pd.DataFrame], pd.Series] = inner) -> pd.Series:
                series = _inner(df)
                if "trade_date" in df.columns:
                    grouped = series.groupby(df["trade_date"])
                    return (series - grouped.transform("mean")) / (grouped.transform("std") + 1e-10)
                return (series - series.mean()) / (series.std() + 1e-10)

            return _cs_zscore

        if func_name in self._UNARY_OPS:
            inner = self._sub_parse(args_str)
            op = self._UNARY_OPS[func_name]
            return lambda df, _op=op, _inner=inner: _op(_inner(df))

        if func_name in self._TS_OPS:
            parts = self._split_top_level(args_str)
            if len(parts) != 2:
                raise ValueError(f"{func_name} requires exactly 2 arguments: (column, window)")
            inner = self._sub_parse(parts[0].strip())
            window = self._parse_window_value(parts[1], func_name)
            op = self._TS_OPS[func_name]
            return lambda df, _op=op, _inner=inner, _w=window: self._apply_ts_op_per_stock(df, _inner, _op, _w)

        if func_name in self._TS_DUAL_OPS:
            parts = self._split_top_level(args_str)
            if len(parts) != 3:
                raise ValueError(f"{func_name} requires exactly 3 arguments: (column1, column2, window)")
            inner1 = self._sub_parse(parts[0].strip())
            inner2 = self._sub_parse(parts[1].strip())
            window = self._parse_window_value(parts[2], func_name)
            op = self._TS_DUAL_OPS[func_name]

            def _ts_dual(df: pd.DataFrame, _op=op, _i1=inner1, _i2=inner2, _w=window) -> pd.Series:
                series1, series2 = _i1(df), _i2(df)
                if "stock_code" in df.columns:
                    temp = pd.DataFrame(
                        {"s1": series1, "s2": series2, "stock_code": df["stock_code"]},
                        index=df.index,
                    )
                    values = temp.groupby("stock_code")[["s1", "s2"]].apply(
                        lambda group: _op(group["s1"], group["s2"], _w)
                    )
                    if isinstance(values.index, pd.MultiIndex):
                        values.index = values.index.get_level_values(-1)
                    return values.reindex(df.index)
                return _op(series1, series2, _w)

            return _ts_dual

        if func_name in self._BINARY_OPS:
            parts = self._split_top_level(args_str)
            if len(parts) != 2:
                raise ValueError(f"{func_name} requires exactly 2 arguments")
            left = self._sub_parse(parts[0].strip())
            right = self._sub_parse(parts[1].strip())
            op = self._BINARY_OPS[func_name]
            return lambda df, _op=op, _left=left, _right=right: _op(_left(df), _right(df))

        if func_name == "trade_when":
            parts = self._split_top_level(args_str)
            if len(parts) != 3:
                raise ValueError("trade_when requires 3 arguments: (condition, alpha, hold_value)")
            cond_fn = self._sub_parse(parts[0].strip())
            alpha_fn = self._sub_parse(parts[1].strip())
            hold_value = float(parts[2].strip())

            def _trade_when(df: pd.DataFrame, _cond=cond_fn, _alpha=alpha_fn, _hold=hold_value) -> pd.Series:
                cond = _cond(df).astype(bool)
                alpha = _alpha(df)
                result = pd.Series(np.nan, index=df.index)
                if "stock_code" in df.columns:
                    for _, group in df.groupby("stock_code"):
                        idx = group.index
                        prev = _hold
                        for row_idx in idx:
                            if cond.loc[row_idx]:
                                prev = alpha.loc[row_idx]
                            result.loc[row_idx] = prev
                else:
                    prev = _hold
                    for row_idx in df.index:
                        if cond.loc[row_idx]:
                            prev = alpha.loc[row_idx]
                        result.loc[row_idx] = prev
                return result

            return _trade_when

        if func_name in ("group_rank", "group_zscore"):
            parts = self._split_top_level(args_str)
            if len(parts) != 2:
                raise ValueError(f"{func_name} requires 2 arguments: (expression, group_column)")
            inner = self._sub_parse(parts[0].strip())
            group_col = parts[1].strip().strip("'\"")
            if func_name == "group_rank":
                def _group_rank(df: pd.DataFrame, _inner=inner, _gc=group_col) -> pd.Series:
                    series = _inner(df)
                    if _gc not in df.columns:
                        return series.groupby(df["trade_date"]).rank(pct=True) if "trade_date" in df.columns else series.rank(pct=True)
                    if "trade_date" in df.columns:
                        return series.groupby([df["trade_date"], df[_gc]]).rank(pct=True)
                    return series.groupby(df[_gc]).rank(pct=True)
                return _group_rank

            def _group_zscore(df: pd.DataFrame, _inner=inner, _gc=group_col) -> pd.Series:
                series = _inner(df)
                if _gc not in df.columns:
                    if "trade_date" in df.columns:
                        grouped = series.groupby(df["trade_date"])
                        return (series - grouped.transform("mean")) / (grouped.transform("std") + 1e-10)
                    return (series - series.mean()) / (series.std() + 1e-10)
                grouped = series.groupby([df["trade_date"], df[_gc]]) if "trade_date" in df.columns else series.groupby(df[_gc])
                return (series - grouped.transform("mean")) / (grouped.transform("std") + 1e-10)

            return _group_zscore

        if func_name == "atr":
            parts = self._split_top_level(args_str)
            if len(parts) != 1:
                raise ValueError("atr requires exactly 1 argument: (window)")
            window = self._parse_window_value(parts[0], func_name)

            def _atr(df: pd.DataFrame, _w=window) -> pd.Series:
                if "stock_code" in df.columns:
                    return df.groupby("stock_code", group_keys=False).apply(lambda group: self._calc_atr(group, _w))
                return self._calc_atr(df, _w)

            return _atr

        if func_name in ("boll_upper", "boll_lower", "boll_mid"):
            parts = self._split_top_level(args_str)
            if len(parts) != 2:
                raise ValueError(f"{func_name} requires exactly 2 arguments: (column, window)")
            inner = self._sub_parse(parts[0].strip())
            window = self._parse_window_value(parts[1], func_name)
            if func_name == "boll_upper":
                return lambda df, _inner=inner, _w=window: _inner(df).rolling(_w, min_periods=1).mean() + 2 * _inner(df).rolling(_w, min_periods=1).std()
            if func_name == "boll_lower":
                return lambda df, _inner=inner, _w=window: _inner(df).rolling(_w, min_periods=1).mean() - 2 * _inner(df).rolling(_w, min_periods=1).std()
            return lambda df, _inner=inner, _w=window: _inner(df).rolling(_w, min_periods=1).mean()

        if func_name == "clip":
            parts = self._split_top_level(args_str)
            if len(parts) != 3:
                raise ValueError("clip requires exactly 3 arguments: (expr, lower, upper)")
            inner = self._sub_parse(parts[0].strip())
            lower = self._sub_parse(parts[1].strip())
            upper = self._sub_parse(parts[2].strip())
            return lambda df, _inner=inner, _lower=lower, _upper=upper: _inner(df).clip(lower=_lower(df), upper=_upper(df))

        if func_name == "where":
            parts = self._split_top_level(args_str)
            if len(parts) != 3:
                raise ValueError("where requires exactly 3 arguments: (condition, true_value, false_value)")
            cond = self._sub_parse(parts[0].strip())
            yes = self._sub_parse(parts[1].strip())
            no = self._sub_parse(parts[2].strip())
            return lambda df, _cond=cond, _yes=yes, _no=no: _yes(df).where(_cond(df).astype(bool), _no(df))

        raise ValueError(f"Unknown function: {func_name}")

    def _build_arithmetic(self, expression: str) -> Callable[[pd.DataFrame], pd.Series]:
        expression = expression.strip()
        if " if " in expression and " else " in expression:
            if_pos = self._find_keyword(expression, " if ")
            else_pos = self._find_keyword(expression, " else ")
            if if_pos is not None and else_pos is not None and if_pos < else_pos:
                true_val = self._sub_parse(expression[:if_pos].strip())
                cond = self._sub_parse(expression[if_pos + 4 : else_pos].strip())
                false_val = self._sub_parse(expression[else_pos + 6 :].strip())
                return lambda df, _t=true_val, _c=cond, _f=false_val: _t(df).where(_c(df) > 0, _f(df))

        for op_str, op_fn in [
            (" or ", lambda a, b: ((a.astype(bool)) | (b.astype(bool))).astype(float)),
            (" and ", lambda a, b: ((a.astype(bool)) & (b.astype(bool))).astype(float)),
        ]:
            index = self._find_keyword(expression, op_str)
            if index is not None:
                left = self._sub_parse(expression[:index])
                right = self._sub_parse(expression[index + len(op_str) :])
                return lambda df, _left=left, _right=right, _op=op_fn: _op(_left(df), _right(df))

        for op_str, op_fn in [
            ("|", lambda a, b: ((a.astype(bool)) | (b.astype(bool))).astype(float)),
            ("&", lambda a, b: ((a.astype(bool)) & (b.astype(bool))).astype(float)),
        ]:
            index = self._find_operator(expression, op_str)
            if index is not None:
                left = self._sub_parse(expression[:index])
                right = self._sub_parse(expression[index + len(op_str) :])
                return lambda df, _left=left, _right=right, _op=op_fn: _op(_left(df), _right(df))

        for op_str, op_fn in [
            (">=", lambda a, b: (a >= b).astype(float)),
            ("<=", lambda a, b: (a <= b).astype(float)),
            ("==", lambda a, b: (a == b).astype(float)),
            ("!=", lambda a, b: (a != b).astype(float)),
            (">", lambda a, b: (a > b).astype(float)),
            ("<", lambda a, b: (a < b).astype(float)),
        ]:
            index = self._find_operator(expression, op_str)
            if index is not None:
                left = self._sub_parse(expression[:index])
                right = self._sub_parse(expression[index + len(op_str) :])
                return lambda df, _left=left, _right=right, _op=op_fn: _op(_left(df), _right(df))

        for op_char, op_fn in [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
            ("/", lambda a, b: a / b.replace(0, np.nan)),
            ("^", lambda a, b: a ** b),
        ]:
            index = self._find_operator(expression, op_char)
            if index is not None:
                left = self._sub_parse(expression[:index])
                right = self._sub_parse(expression[index + 1 :])
                return lambda df, _left=left, _right=right, _op=op_fn: _op(_left(df), _right(df))

        if expression.startswith("-"):
            inner = self._sub_parse(expression[1:])
            return lambda df, _inner=inner: -_inner(df)

        if expression.startswith("(") and expression.endswith(")"):
            return self._sub_parse(expression[1:-1])

        try:
            value = float(expression)
            return lambda df, _value=value: pd.Series(_value, index=df.index)
        except ValueError:
            pass

        expr_lower = expression.lower()
        if expr_lower in self._SPECIAL_VARS:
            fn = self._SPECIAL_VARS[expr_lower]
            return lambda df, _fn=fn: _fn(df)

        if expr_lower.startswith("adv") and expr_lower[3:].isdigit():
            window = self._validate_window(int(expr_lower[3:]), "adv")
            return lambda df, _w=window: df.groupby("stock_code")["volume"].transform(lambda x: x.rolling(_w, min_periods=1).mean()) if "stock_code" in df.columns else df["volume"].rolling(_w, min_periods=1).mean()

        col_name = {
            "pe_ratio": "pe",
            "pe_ttm": "pe",
            "pb_ratio": "pb",
            "ps_ratio": "ps",
            "eps": "eps_ttm",
            "roe_avg": "roe",
            "div_yield": "dividend_yield",
        }.get(expr_lower.strip(), expr_lower.strip())
        if col_name not in ALLOWED_COLUMNS:
            raise ValueError(f"Unknown column or variable: {col_name!r}")
        return lambda df, _column=col_name: df[_column]

    @staticmethod
    def _find_keyword(expr: str, keyword: str) -> int | None:
        depth = 0
        result = None
        keyword_len = len(keyword)
        for index in range(len(expr) - keyword_len + 1):
            char = expr[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and expr[index : index + keyword_len] == keyword:
                result = index
        return result

    @staticmethod
    def _find_operator(expr: str, op: str) -> int | None:
        depth = 0
        result = None
        op_len = len(op)
        index = 0
        while index < len(expr):
            char = expr[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and index > 0 and expr[index : index + op_len] == op:
                if op_len == 1 and char in "+-":
                    prev_char = expr[index - 1] if index > 0 else ""
                    next_char = expr[index + 1] if index + 1 < len(expr) else ""
                    if prev_char.lower() == "e" and (next_char.isdigit() or next_char == "."):
                        index += 1
                        continue
                if op_len == 1 and char in "<>=!":
                    next_char = expr[index + 1] if index + 1 < len(expr) else ""
                    prev_char = expr[index - 1] if index > 0 else ""
                    if next_char == "=" or (char == "=" and prev_char in "<>!="):
                        index += 1
                        continue
                result = index
            index += 1
        return result

    @staticmethod
    def _split_top_level(value: str) -> list[str]:
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for char in value:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current))
        return parts

    @staticmethod
    def _convert_ternary_operators(expression: str) -> str:
        max_iterations = 20
        pattern = r"\(([^()]+)\)\s*\?\s*([^:]+?)\s*:\s*([^)]+?)(?=\))"
        iteration = 0
        while "?" in expression and iteration < max_iterations:
            iteration += 1
            old_expression = expression

            def replace_ternary(match: re.Match[str]) -> str:
                condition = match.group(1).strip()
                true_val = match.group(2).strip()
                false_val = match.group(3).strip()
                return f"({true_val} if {condition} else {false_val})"

            expression = re.sub(pattern, replace_ternary, expression, count=1)
            if expression == old_expression:
                break
        return expression


@dataclass
class QuantGPTExecutionArtifacts:
    canonical_expression: str
    canonical_ast: dict[str, Any]
    evaluator: Callable[[pd.DataFrame], pd.Series]


class QuantGPTExpressionEngine:
    """将 QuantGPT 风格表达式执行链封装到 FactorHub。"""

    def compile_expression(self, expression: str) -> QuantGPTExecutionArtifacts:
        parser = ExpressionParser()
        adapted_expression = ExpressionAdapter.adapt(expression)
        evaluator = parser.parse(adapted_expression)
        canonical_expression = normalize_expression(adapted_expression)
        canonical_ast = extract_components(adapted_expression)
        return QuantGPTExecutionArtifacts(
            canonical_expression=canonical_expression,
            canonical_ast=canonical_ast,
            evaluator=evaluator,
        )

    def detect_needed_vars(self, expression: str) -> set[str]:
        adapted_expression = ExpressionAdapter.adapt(expression)
        return market_data_enrichment_service.detect_variables([adapted_expression])

    def build_panel_data(
        self,
        *,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        expression: str,
        stock_data_loader: Callable[[str, str, str], pd.DataFrame | None],
    ) -> pd.DataFrame:
        needed_vars = self.detect_needed_vars(expression)
        frames: list[pd.DataFrame] = []
        for stock_code in stock_codes:
            market_df = stock_data_loader(stock_code, start_date, end_date)
            if market_df is None or market_df.empty:
                continue
            enriched = market_data_enrichment_service.enrich_daily_data(
                market_df=market_df.copy(),
                stock_code=stock_code,
                start_date=start_date,
                end_date=end_date,
                needed_vars=needed_vars,
            )
            frame = self._normalize_market_frame(enriched, stock_code=stock_code)
            if frame.empty:
                continue
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        panel = pd.concat(frames, ignore_index=True)
        panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
        panel = panel.dropna(subset=["trade_date"]).sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
        return panel

    def execute_on_panel(self, panel_df: pd.DataFrame, expression: str) -> EngineExecutionResult:
        compiled = self.compile_expression(expression)
        factor_series = compiled.evaluator(panel_df)
        if not isinstance(factor_series, pd.Series):
            factor_series = pd.Series(factor_series, index=panel_df.index)
        factor_series = pd.to_numeric(factor_series, errors="coerce")
        return EngineExecutionResult(
            raw_expression=expression,
            engine_type="quantgpt",
            dialect="quantgpt_local",
            factor_series=factor_series,
            metrics_source="quantgpt_expression_engine",
            diagnostics=[],
            canonical_expression=compiled.canonical_expression,
            canonical_ast=compiled.canonical_ast,
            execution_meta={
                "component_fields": compiled.canonical_ast.get("fields", []),
                "component_operators": compiled.canonical_ast.get("operators", []),
                "panel_rows": int(len(panel_df)),
                "stock_count": int(panel_df["stock_code"].nunique()) if "stock_code" in panel_df.columns else 1,
            },
        )

    def can_execute_on_frames(self, expression: str, sample_frames: list[pd.DataFrame]) -> bool:
        if not sample_frames:
            return False
        normalized_frames = [
            self._normalize_market_frame(frame.copy(), stock_code=f"SAMPLE_{index:03d}")
            for index, frame in enumerate(sample_frames, start=1)
        ]
        panel_df = pd.concat([frame for frame in normalized_frames if not frame.empty], ignore_index=True)
        if panel_df.empty:
            return False
        result = self.execute_on_panel(panel_df, expression)
        return int(result.factor_series.dropna().shape[0]) > 0 if result.factor_series is not None else False

    def _normalize_market_frame(self, df: pd.DataFrame, *, stock_code: str) -> pd.DataFrame:
        frame = df.copy()
        if "trade_date" not in frame.columns:
            if "date" in frame.columns:
                frame["trade_date"] = pd.to_datetime(frame["date"], errors="coerce")
            elif frame.index.name in {"date", "trade_date"} or isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.reset_index()
                if "trade_date" in frame.columns:
                    source_col = "trade_date"
                elif "date" in frame.columns:
                    source_col = "date"
                else:
                    source_col = frame.columns[0]
                frame["trade_date"] = pd.to_datetime(frame[source_col], errors="coerce")
            else:
                frame["trade_date"] = pd.to_datetime(frame.index, errors="coerce")
        else:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")

        if "stock_code" not in frame.columns:
            frame["stock_code"] = stock_code
        frame["stock_code"] = frame["stock_code"].astype(str)

        if "close" in frame.columns:
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        for column in ("open", "high", "low", "volume", "amount", "pct_change", "market_cap", "float_market_cap", "turnover_rate"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "vwap" not in frame.columns and {"amount", "volume"} <= set(frame.columns):
            frame["vwap"] = frame["amount"] / frame["volume"].replace(0, np.nan)
        return frame
