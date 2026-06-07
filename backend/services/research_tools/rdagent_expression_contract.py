from __future__ import annotations

import re

from backend.services.research_tools.expression_adapter import ExpressionAdapter


RDAGENT_ALLOWED_EXPRESSION_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "pct_change",
}

RDAGENT_ALLOWED_EXPRESSION_CONSTANTS = {
    "nan",
    "inf",
}

RDAGENT_UNARY_FUNCTIONS = {
    "rank",
    "tanh",
    "log",
    "abs",
    "sigmoid",
    "sign",
    "sqrt",
    "exp",
}

RDAGENT_WINDOW_FUNCTIONS = {
    "ts_mean",
    "ts_std",
    "ts_zscore",
    "decay_linear",
    "ts_min",
    "ts_max",
    "ts_shift",
    "ts_delta",
}

RDAGENT_PAIR_WINDOW_FUNCTIONS = {
    "ts_corr",
    "ts_cov",
}

RDAGENT_BINARY_FUNCTIONS = {
    "min",
    "max",
}

RDAGENT_TERNARY_FUNCTIONS = {
    "where",
}

RDAGENT_ALLOWED_EXPRESSION_FUNCTIONS = (
    RDAGENT_UNARY_FUNCTIONS
    | RDAGENT_WINDOW_FUNCTIONS
    | RDAGENT_PAIR_WINDOW_FUNCTIONS
    | RDAGENT_BINARY_FUNCTIONS
    | RDAGENT_TERNARY_FUNCTIONS
)


def canonicalize_factor_code(raw_code: str) -> str:
    expr = str(raw_code or "").strip()
    if not expr:
        return expr
    lines = [line.strip() for line in expr.splitlines() if line.strip()]
    if len(lines) == 1:
        return lines[0]
    return " ".join(lines)


def rdagent_expression_contract_text() -> str:
    return (
        "FactorHub 表达式契约：formulation 必须是单行表达式，不允许 Python/SQL/自然语言；"
        "字段只使用 open, high, low, close, volume, amount, vwap, pct_change；"
        "推荐 2 到 4 个简单算子组合，例如 rank(ts_delta(close, 5)) 或 "
        "rank(ts_mean(volume, 10) / (ts_std(volume, 10) + 1e-6))，也可以使用 "
        "ts_zscore(close, 20) 这类时序标准化；"
        "rank/tanh/log/abs/sigmoid 只能传 1 个参数，sign/sqrt/exp 也只能传 1 个参数；"
        "ts_mean/ts_std/ts_zscore/decay_linear/ts_min/ts_max 必须传 (表达式, 整数窗口)，"
        "ts_shift/ts_delta 也必须传 (表达式, 整数窗口)；"
        "ts_corr/ts_cov 必须传 (表达式1, 表达式2, 整数窗口)；"
        "比较两个序列大小必须使用 min(x, y) / max(x, y)，不要写 ts_min(x, y) 或 ts_max(x, y)；"
        "不要把窗口参数塞给一元函数，例如不要写 tanh(x, 5)；"
        "不要反复内联同一个长子式，避免重复内联很长子式，整体函数嵌套深度必须显著低于 100。"
    )


def normalize_rdagent_expression_for_parser(expression: str) -> str:
    expr = str(expression or "")
    expr = ExpressionAdapter.adapt(expr)
    expr = canonicalize_factor_code(expr)
    return expr.strip()


class RDAgentExpressionFormatError(RuntimeError):
    """Raised when an RDAgent expression violates the FactorHub expression contract."""


def validate_rdagent_expression_contract(expression: str) -> None:
    expr = str(expression or "").strip()
    if not expr:
        raise RDAgentExpressionFormatError("RDAgent 表达式为空")
    if "\n" in expr or "\r" in expr:
        raise RDAgentExpressionFormatError("RDAgent formulation 必须是单行 FactorHub 表达式，不能包含多行代码")

    lowered = expr.lower()
    if lowered.startswith(("def ", "class ", "import ", "from ", "select ", "with ")):
        raise RDAgentExpressionFormatError("RDAgent formulation 必须是表达式，不能是 Python/SQL 代码")
    if re.search(r"\b(return|lambda|for|while|if|elif|else|try|except|yield)\b", expr):
        raise RDAgentExpressionFormatError("RDAgent formulation 不能包含 Python 语句或控制流")
    if re.search(r"(^|[^=!<>])=(?!=)", expr):
        raise RDAgentExpressionFormatError("RDAgent formulation 不能包含赋值语句")
    if any(token in expr for token in ("```", ";")):
        raise RDAgentExpressionFormatError("RDAgent formulation 不能包含 markdown 代码块或语句分隔符")
    if re.search(r"[\u4e00-\u9fff]", expr):
        raise RDAgentExpressionFormatError("RDAgent formulation 不能包含自然语言说明")

    function_names = {
        match.group(1).lower()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr)
    }
    unknown_functions = sorted(
        name for name in function_names if name not in RDAGENT_ALLOWED_EXPRESSION_FUNCTIONS
    )
    if unknown_functions:
        raise RDAgentExpressionFormatError(
            "RDAgent formulation 使用了 FactorHub 不支持的函数: "
            + ", ".join(unknown_functions)
        )

    identifiers = {
        match.group(0).lower()
        for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr)
    }
    unknown_identifiers = sorted(
        name
        for name in identifiers
        if name not in RDAGENT_ALLOWED_EXPRESSION_FIELDS
        and name not in RDAGENT_ALLOWED_EXPRESSION_FUNCTIONS
        and name not in RDAGENT_ALLOWED_EXPRESSION_CONSTANTS
    )
    if unknown_identifiers:
        raise RDAgentExpressionFormatError(
            "RDAgent formulation 使用了 FactorHub 不支持的字段或变量: "
            + ", ".join(unknown_identifiers)
        )

    _validate_supported_function_arities(expr)


def _validate_supported_function_arities(expr: str) -> None:
    for function_name, args in _iter_function_calls(expr):
        lowered = function_name.lower()

        if lowered in RDAGENT_UNARY_FUNCTIONS:
            if len(args) != 1:
                raise RDAgentExpressionFormatError(f"{function_name} 只能接收 1 个参数")
            continue

        if lowered in RDAGENT_WINDOW_FUNCTIONS:
            if len(args) != 2:
                raise RDAgentExpressionFormatError(f"{function_name} 必须接收 2 个参数: (表达式, 整数窗口)")
            if not ExpressionAdapter._is_integer_literal(args[1]):
                raise RDAgentExpressionFormatError(f"{function_name} 的窗口参数必须是整数")
            continue

        if lowered in RDAGENT_PAIR_WINDOW_FUNCTIONS:
            if len(args) != 3:
                raise RDAgentExpressionFormatError(f"{function_name} 必须接收 3 个参数: (表达式1, 表达式2, 整数窗口)")
            if not ExpressionAdapter._is_integer_literal(args[2]):
                raise RDAgentExpressionFormatError(f"{function_name} 的窗口参数必须是整数")
            continue

        if lowered in RDAGENT_BINARY_FUNCTIONS:
            if len(args) != 2:
                raise RDAgentExpressionFormatError(f"{function_name} 必须接收 2 个参数")
            continue

        if lowered in RDAGENT_TERNARY_FUNCTIONS and len(args) != 3:
            raise RDAgentExpressionFormatError(f"{function_name} 必须接收 3 个参数")


def _iter_function_calls(expr: str) -> list[tuple[str, list[str]]]:
    calls: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(expr):
        if index > 0 and (expr[index - 1].isalnum() or expr[index - 1] == "_"):
            index += 1
            continue

        match = re.match(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", expr[index:])
        if not match:
            index += 1
            continue

        function_name = match.group(1)
        open_idx = index + match.end() - 1
        close_idx = ExpressionAdapter._find_matching_paren(expr, open_idx)
        if close_idx is None:
            raise RDAgentExpressionFormatError(f"{function_name} 括号未闭合")

        args = ExpressionAdapter._split_top_level(expr[open_idx + 1:close_idx])
        calls.append((function_name, args))
        for arg in args:
            calls.extend(_iter_function_calls(arg))
        index = close_idx + 1
    return calls


__all__ = [
    "RDAGENT_ALLOWED_EXPRESSION_FIELDS",
    "RDAGENT_ALLOWED_EXPRESSION_FUNCTIONS",
    "RDAGENT_ALLOWED_EXPRESSION_CONSTANTS",
    "RDAgentExpressionFormatError",
    "canonicalize_factor_code",
    "normalize_rdagent_expression_for_parser",
    "rdagent_expression_contract_text",
    "validate_rdagent_expression_contract",
]
