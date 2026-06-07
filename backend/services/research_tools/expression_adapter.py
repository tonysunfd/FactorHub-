from __future__ import annotations

import re


class ExpressionAdapter:
    """Translate generator dialects into FactorHub expression-engine syntax.

    This is intentionally conservative: only rewrite patterns we have high confidence in.
    Unknown expressions pass through unchanged.
    """

    GENERATOR_DIALECT_DIFFS = {
        "Ref(x,N)": "ts_shift(x,N)",
        "Mean(x,N)": "ts_mean(x,N)",
        "Std(x,N)": "ts_std(x,N)",
        "If(c,t,f)": "where(c,t,f)",
        "$close/$volume": "close/volume",
        "Corr/Cov": "ts_corr/ts_cov",
        "Abs/Log/Max/Min": "abs/log/max/min",
        "div(a,b)": "a / b",
        "turnover": "volume / ts_mean(volume,20)",
        "turnover_rate": "volume / adv20",
        "pandas shift/pct_change/rolling": "ts_shift/ts_delta/ts_mean/ts_std",
        "df['close']/df[\"close\"]": "close",
    }

    @classmethod
    def adapt(cls, expression: str) -> str:
        expr = (expression or "").strip()
        if not expr:
            return expr

        expr = cls._replace_np_log(expr)
        expr = cls._replace_dataframe_column_refs(expr)
        expr = cls._replace_shift(expr)
        expr = cls._replace_pct_change(expr)
        expr = cls._replace_rolling_mean(expr)
        expr = cls._replace_rolling_std(expr)
        expr = cls._replace_sma(expr)
        expr = cls._replace_hhv_llv(expr)
        expr = cls._replace_qlib_field_prefix(expr)
        expr = cls._replace_factorhub_function_aliases(expr)
        expr = cls._replace_misused_time_series_extrema(expr)
        expr = cls._replace_misused_unary_trailing_window(expr)
        expr = cls._replace_function_name_case(expr)
        expr = cls._replace_field_aliases(expr)
        return expr

    @staticmethod
    def _replace_np_log(expr: str) -> str:
        expr = re.sub(r"\bnp\.log\s*\(", "log(", expr)
        expr = re.sub(r"\bnumpy\.log\s*\(", "log(", expr)
        return expr

    @staticmethod
    def _replace_shift(expr: str) -> str:
        return re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)\.shift\((\d+)\)", r"ts_shift(\1,\2)", expr)

    @staticmethod
    def _replace_pct_change(expr: str) -> str:
        expr = re.sub(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.pct_change\((\d+)\)",
            r"(\1 / ts_shift(\1,\2) - 1)",
            expr,
        )
        expr = re.sub(r"\bpct_change\b", "returns", expr, flags=re.IGNORECASE)
        return expr

    @staticmethod
    def _replace_rolling_mean(expr: str) -> str:
        return re.sub(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.rolling\(window\s*=\s*(\d+)\)\.mean\(\)",
            r"ts_mean(\1,\2)",
            expr,
        )

    @staticmethod
    def _replace_rolling_std(expr: str) -> str:
        return re.sub(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.rolling\(window\s*=\s*(\d+)\)\.std\(\)",
            r"ts_std(\1,\2)",
            expr,
        )

    @staticmethod
    def _replace_sma(expr: str) -> str:
        expr = re.sub(r"\bSMA\(([^,]+),\s*timeperiod\s*=\s*(\d+)\)", r"ts_mean(\1,\2)", expr)
        expr = re.sub(r"\bSMA\(([^,]+),\s*(\d+)\)", r"ts_mean(\1,\2)", expr)
        expr = re.sub(r"\bMA\(([^,]+),\s*(\d+)\)", r"ts_mean(\1,\2)", expr)
        return expr

    @staticmethod
    def _replace_hhv_llv(expr: str) -> str:
        expr = re.sub(r"\bHHV\(([^,]+),\s*(\d+)\)", r"ts_max(\1,\2)", expr)
        expr = re.sub(r"\bLLV\(([^,]+),\s*(\d+)\)", r"ts_min(\1,\2)", expr)
        return expr

    @classmethod
    def _replace_factorhub_function_aliases(cls, expr: str) -> str:
        for source, target in [
            ("Ref", "ts_shift"),
            ("Delay", "ts_shift"),
            ("delta", "ts_delta"),
            ("ts_delay", "ts_shift"),
            ("Mean", "ts_mean"),
            ("Std", "ts_std"),
            ("stddev", "ts_std"),
            ("Sum", "ts_sum"),
            ("Corr", "ts_corr"),
            ("Cov", "ts_cov"),
            ("Covariance", "ts_cov"),
            ("Correlation", "ts_corr"),
            ("Max", "max"),
            ("Min", "min"),
            ("If", "where"),
        ]:
            expr = re.sub(rf"\b{source}\s*\(", f"{target}(", expr, flags=re.IGNORECASE)
        expr = cls._replace_div(expr)
        return expr

    @staticmethod
    def _replace_qlib_field_prefix(expr: str) -> str:
        return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"\1", expr)

    @staticmethod
    def _replace_dataframe_column_refs(expr: str) -> str:
        return re.sub(r"\bdf\s*\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]", r"\1", expr)

    @staticmethod
    def _replace_function_name_case(expr: str) -> str:
        for source, target in [
            ("Abs", "abs"),
            ("Log", "log"),
            ("Rank", "rank"),
            ("ZScore", "zscore"),
            ("Sign", "sign"),
            ("Sqrt", "sqrt"),
            ("Power", "power"),
        ]:
            expr = re.sub(rf"\b{source}\s*\(", f"{target}(", expr, flags=re.IGNORECASE)
        return expr

    @classmethod
    def _replace_div(cls, expr: str) -> str:
        return cls._replace_function_call(expr, "div", lambda args: f"{args[0]} / ({args[1]})" if len(args) == 2 else None)

    @classmethod
    def _replace_misused_time_series_extrema(cls, expr: str) -> str:
        def rewrite(function_name: str, binary_name: str, current: str) -> str:
            return cls._replace_function_call(
                current,
                function_name,
                lambda args: (
                    f"{binary_name}({args[0]}, {args[1]})"
                    if len(args) == 2 and not cls._is_integer_literal(args[1])
                    else None
                ),
            )

        expr = rewrite("ts_min", "min", expr)
        expr = rewrite("ts_max", "max", expr)
        return expr

    @classmethod
    def _replace_misused_unary_trailing_window(cls, expr: str) -> str:
        for function_name in ("rank", "zscore", "tanh", "log", "abs", "sigmoid", "sign", "sqrt", "exp"):
            expr = cls._replace_function_call(
                expr,
                function_name,
                lambda args: (
                    f"{function_name}({args[0]})"
                    if len(args) == 2 and cls._is_integer_literal(args[1])
                    else None
                ),
            )
        return expr

    @staticmethod
    def _is_integer_literal(value: str) -> bool:
        return bool(re.fullmatch(r"[+-]?\d+", str(value or "").strip()))

    @staticmethod
    def _replace_field_aliases(expr: str) -> str:
        expr = re.sub(r"\bturnover_rate\b", "volume / adv20", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bturnover\b", "volume / ts_mean(volume,20)", expr, flags=re.IGNORECASE)
        return expr

    @classmethod
    def _replace_function_call(cls, expr: str, function_name: str, replacement) -> str:
        name = function_name.lower()
        i = 0
        out: list[str] = []
        while i < len(expr):
            if i > 0 and (expr[i - 1].isalnum() or expr[i - 1] == "_"):
                out.append(expr[i])
                i += 1
                continue

            match = re.match(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr[i:])
            if not match or match.group(1).lower() != name:
                out.append(expr[i])
                i += 1
                continue

            name_start = i
            open_idx = i + match.end() - 1
            close_idx = cls._find_matching_paren(expr, open_idx)
            if close_idx is None:
                out.append(expr[i])
                i += 1
                continue

            args = cls._split_top_level(expr[open_idx + 1:close_idx])
            rewritten = replacement(args)
            if rewritten is None:
                out.append(expr[name_start:close_idx + 1])
            else:
                out.append(rewritten)
            i = close_idx + 1
        return "".join(out)

    @staticmethod
    def _find_matching_paren(expr: str, open_idx: int) -> int | None:
        depth = 0
        for idx in range(open_idx, len(expr)):
            char = expr[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return idx
        return None

    @staticmethod
    def _split_top_level(args: str) -> list[str]:
        parts: list[str] = []
        depth = 0
        start = 0
        for idx, char in enumerate(args):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(args[start:idx].strip())
                start = idx + 1
        parts.append(args[start:].strip())
        return parts
