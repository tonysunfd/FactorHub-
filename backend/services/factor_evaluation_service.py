from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from backend.services.expression_schema import EngineExecutionResult, FactorEvaluationResult


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _grade_from_score(score: float) -> str:
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 40:
        return "C"
    return "D"


class FactorEvaluationService:
    """统一的因子评价服务。"""

    def _build_panel_from_stock_rows(
        self,
        *,
        expression: str,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        holding_period: int,
        stock_data_loader: Callable[[str, str, str], pd.DataFrame | None],
        expression_executor: Callable[[pd.DataFrame, str], pd.Series | Any],
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        rows: list[pd.DataFrame] = []
        diagnostics: list[dict[str, Any]] = []

        for stock_code in stock_codes:
            try:
                stock_df = stock_data_loader(stock_code, start_date, end_date)
                if stock_df is None or len(stock_df) == 0:
                    continue
                factor_values = expression_executor(stock_df.copy(), expression)
                if factor_values is None:
                    continue

                factor_series = factor_values if isinstance(factor_values, pd.Series) else pd.Series(factor_values)
                date_index = stock_df.index if getattr(stock_df.index, "dtype", None) is not None else stock_df.get("date")
                future_return = stock_df["close"].pct_change(holding_period).shift(-holding_period)
                frame = pd.DataFrame(
                    {
                        "date": pd.to_datetime(date_index),
                        "stock_code": stock_code,
                        "factor": factor_series,
                        "future_return": future_return,
                    }
                ).dropna()
                if len(frame) < 20:
                    continue
                rows.append(frame)
            except Exception as exc:
                diagnostics.append({"type": "warning", "label": "单票评估失败", "text": f"{stock_code} 评估失败：{exc}"})

        if not rows:
            return pd.DataFrame(), diagnostics

        return pd.concat(rows, ignore_index=True), diagnostics

    def _evaluate_panel(
        self,
        *,
        panel: pd.DataFrame,
        expression: str,
        prompt: str,
        benchmark: str,
        n_groups: int,
        holding_period: int,
        direction: str,
        benchmark_loader: Callable[[str, str, str], pd.Series | None],
        report_writer: Callable[[pd.Series, pd.Series | None, int], tuple[dict[str, Any], str]],
        engine_type: str,
        dialect: str,
        canonical_expression: str | None,
        canonical_ast: dict[str, Any] | None,
        diagnostics: list[dict[str, Any]] | None = None,
        start_date: str = "",
        end_date: str = "",
        metrics_source: str = "factor_evaluation_service",
        execution_meta: dict[str, Any] | None = None,
    ) -> FactorEvaluationResult | None:
        if panel.empty:
            return None

        diagnostics = list(diagnostics or [])
        clean_panel = panel.copy()
        clean_panel["date"] = pd.to_datetime(clean_panel["date"], errors="coerce")
        clean_panel["factor"] = pd.to_numeric(clean_panel["factor"], errors="coerce")
        clean_panel["future_return"] = pd.to_numeric(clean_panel["future_return"], errors="coerce")
        clean_panel = clean_panel.dropna(subset=["date", "stock_code", "factor", "future_return"])
        if clean_panel.empty:
            return None

        grouped = clean_panel.groupby("date")
        daily_rank_ic: list[float] = []
        daily_spread: list[float] = []
        daily_dates: list[pd.Timestamp] = []
        top_memberships: list[set[str]] = []

        for trade_date, group in grouped:
            clean = group[["stock_code", "factor", "future_return"]].dropna()
            if len(clean) < max(n_groups * 2, 8):
                continue

            rank_ic = clean["factor"].corr(clean["future_return"], method="spearman")
            if pd.notna(rank_ic):
                daily_rank_ic.append(float(rank_ic))

            ranked = clean.assign(rank=clean["factor"].rank(method="first"))
            try:
                quantiles = pd.qcut(ranked["rank"], q=n_groups, labels=False, duplicates="drop")
            except ValueError:
                continue
            ranked = ranked.assign(quantile=quantiles)
            if ranked["quantile"].nunique() < 2:
                continue

            grouped_return = ranked.groupby("quantile")["future_return"].mean()
            spread = grouped_return.iloc[-1] - grouped_return.iloc[0]
            daily_spread.append(float(spread))
            daily_dates.append(pd.Timestamp(trade_date))

            top_quantile = ranked["quantile"].max()
            top_memberships.append(set(ranked.loc[ranked["quantile"] == top_quantile, "stock_code"].tolist()))

        if not daily_rank_ic or not daily_spread:
            return None

        spread_series = pd.Series(daily_spread, index=pd.to_datetime(daily_dates), dtype=float).sort_index()
        rank_ic_series = pd.Series(daily_rank_ic, index=pd.to_datetime(daily_dates), dtype=float).sort_index()

        periods_per_year = max(1.0, 252.0 / max(holding_period, 1))
        rank_ic_mean = _safe_float(rank_ic_series.mean())
        rank_ic_std = _safe_float(rank_ic_series.std())
        ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-12 else 0.0
        ic_win_rate = _safe_float((rank_ic_series > 0).mean())

        spread_mean = _safe_float(spread_series.mean())
        spread_std = _safe_float(spread_series.std())
        long_short_sharpe = spread_mean / spread_std * math.sqrt(periods_per_year) if spread_std > 1e-12 else 0.0
        long_short_annual = spread_mean * periods_per_year

        cumulative = (1.0 + spread_series.fillna(0.0)).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1.0
        max_drawdown = abs(_safe_float(drawdown.min()))

        turnover_values: list[float] = []
        for index in range(1, len(top_memberships)):
            previous = top_memberships[index - 1]
            current = top_memberships[index]
            union = previous | current
            if not union:
                continue
            overlap = len(previous & current) / len(union)
            turnover_values.append(1.0 - overlap)
        turnover = _safe_float(np.mean(turnover_values) if turnover_values else 0.0)

        fitness_base = abs(long_short_sharpe) * math.sqrt(abs(long_short_annual) / max(turnover, 0.125) + 1e-8)
        wq_fitness = round(_safe_float(fitness_base), 4)

        component_scores = self.compute_component_scores(
            rank_ic_mean=rank_ic_mean,
            ic_ir=ic_ir,
            ic_win_rate=ic_win_rate,
            long_short_sharpe=long_short_sharpe,
            long_short_annual=long_short_annual,
            turnover=turnover,
            wq_fitness=wq_fitness,
            max_drawdown=max_drawdown,
        )
        score = round(min(max(component_scores["total_score"], 0.0), 100.0), 2)
        grade = _grade_from_score(score)

        benchmark_returns = benchmark_loader(benchmark, start_date, end_date)
        report_metrics, report_url = report_writer(spread_series, benchmark_returns, int(periods_per_year))
        backtest_summary = {
            "long_short_sharpe": round(_safe_float(long_short_sharpe), 4),
            "long_short_annual": round(_safe_float(long_short_annual), 4),
            "top_group_sharpe": round(_safe_float(long_short_sharpe), 4),
            "monotonicity_score": round(float((spread_series > 0).mean()), 4),
            "spread": round(_safe_float(spread_mean), 6),
            "group_returns": {
                "top_minus_bottom_mean": round(_safe_float(spread_mean), 6),
            },
            "rank_ic_mean": round(_safe_float(rank_ic_mean), 6),
            "ic_mean": round(_safe_float(rank_ic_mean), 6),
            "ic_ir": round(_safe_float(ic_ir), 6),
            "ic_win_rate": round(_safe_float(ic_win_rate), 6),
            "turnover": round(_safe_float(turnover), 6),
            "wq_fitness": wq_fitness,
        }
        wq_brain = {
            "wq_rating": grade,
            "wq_fitness": wq_fitness,
            "wq_sharpe": round(_safe_float(long_short_sharpe), 4),
            "wq_returns": round(_safe_float(long_short_annual), 4),
            "wq_turnover": round(_safe_float(turnover), 4),
            "submittable": score >= 60,
        }
        anti_overfit = self.build_anti_overfit_result(
            rank_ic_series=rank_ic_series,
            spread_series=spread_series,
            turnover=turnover,
            max_drawdown=max_drawdown,
        )
        interpretation = self.build_interpretation(
            prompt=prompt,
            report_metrics=report_metrics,
            backtest_summary=backtest_summary,
            wq_brain=wq_brain,
            anti_overfit=anti_overfit,
        )
        diagnostics.insert(
            0,
            {
                "type": "info",
                "label": "评估完成",
                "text": f"使用 {int(clean_panel['stock_code'].nunique())} 只股票、{len(spread_series)} 个截面日期完成真实候选评估。",
            },
        )

        merged_meta = dict(execution_meta or {})
        merged_meta.setdefault("prompt", prompt)
        merged_meta.setdefault("direction", direction)
        merged_meta["stock_count"] = int(clean_panel["stock_code"].nunique())
        merged_meta["date_count"] = len(spread_series)

        execution_result = EngineExecutionResult(
            raw_expression=expression,
            engine_type=engine_type,
            dialect=dialect,
            factor_series=None,
            metrics_source=metrics_source,
            diagnostics=list(diagnostics),
            canonical_expression=canonical_expression,
            canonical_ast=canonical_ast,
            execution_meta=merged_meta,
        )
        return FactorEvaluationResult(
            expression=canonical_expression or expression,
            raw_expression=expression,
            engine_type=engine_type,
            dialect=dialect,
            canonical_expression=canonical_expression,
            canonical_ast=canonical_ast,
            score=score,
            grade=grade,
            report_metrics=report_metrics,
            backtest_summary=backtest_summary,
            wq_brain=wq_brain,
            component_scores=component_scores,
            anti_overfit=anti_overfit,
            interpretation=interpretation,
            diagnostics=execution_result.diagnostics,
            report_url=report_url,
            execution_meta=execution_result.execution_meta,
        )

    def evaluate_factor_expression(
        self,
        *,
        expression: str,
        prompt: str,
        stock_codes: list[str],
        start_date: str,
        end_date: str,
        benchmark: str,
        n_groups: int,
        holding_period: int,
        direction: str,
        stock_data_loader: Callable[[str, str, str], pd.DataFrame | None],
        expression_executor: Callable[[pd.DataFrame, str], pd.Series | Any],
        benchmark_loader: Callable[[str, str, str], pd.Series | None],
        report_writer: Callable[[pd.Series, pd.Series | None, int], tuple[dict[str, Any], str]],
        engine_type: str = "factorhub",
        dialect: str = "factorhub_native",
        canonical_expression: str | None = None,
        canonical_ast: dict[str, Any] | None = None,
    ) -> FactorEvaluationResult | None:
        panel, diagnostics = self._build_panel_from_stock_rows(
            expression=expression,
            stock_codes=stock_codes,
            start_date=start_date,
            end_date=end_date,
            holding_period=holding_period,
            stock_data_loader=stock_data_loader,
            expression_executor=expression_executor,
        )
        return self._evaluate_panel(
            panel=panel,
            expression=expression,
            prompt=prompt,
            benchmark=benchmark,
            n_groups=n_groups,
            holding_period=holding_period,
            direction=direction,
            benchmark_loader=benchmark_loader,
            report_writer=report_writer,
            engine_type=engine_type,
            dialect=dialect,
            canonical_expression=canonical_expression,
            canonical_ast=canonical_ast,
            diagnostics=diagnostics,
            start_date=start_date,
            end_date=end_date,
        )

    def evaluate_factor_panel(
        self,
        *,
        expression: str,
        prompt: str,
        panel_df: pd.DataFrame,
        factor_series: pd.Series,
        benchmark: str,
        start_date: str,
        end_date: str,
        n_groups: int,
        holding_period: int,
        direction: str,
        benchmark_loader: Callable[[str, str, str], pd.Series | None],
        report_writer: Callable[[pd.Series, pd.Series | None, int], tuple[dict[str, Any], str]],
        engine_type: str,
        dialect: str,
        canonical_expression: str | None,
        canonical_ast: dict[str, Any] | None,
        diagnostics: list[dict[str, Any]] | None = None,
        execution_meta: dict[str, Any] | None = None,
        metrics_source: str = "factor_evaluation_service",
    ) -> FactorEvaluationResult | None:
        if panel_df.empty or factor_series is None:
            return None

        clean_panel = panel_df.copy()
        clean_panel["factor"] = pd.to_numeric(
            factor_series.reindex(clean_panel.index) if isinstance(factor_series, pd.Series) else pd.Series(factor_series, index=clean_panel.index),
            errors="coerce",
        )
        close_series = pd.to_numeric(clean_panel["close"], errors="coerce")
        if "stock_code" in clean_panel.columns:
            future_return = close_series.groupby(clean_panel["stock_code"]).pct_change(holding_period).shift(-holding_period)
        else:
            future_return = close_series.pct_change(holding_period).shift(-holding_period)
        clean_panel["future_return"] = future_return
        date_column = "trade_date" if "trade_date" in clean_panel.columns else "date"
        clean_panel["date"] = pd.to_datetime(clean_panel[date_column], errors="coerce")

        return self._evaluate_panel(
            panel=clean_panel[["date", "stock_code", "factor", "future_return"]],
            expression=expression,
            prompt=prompt,
            benchmark=benchmark,
            n_groups=n_groups,
            holding_period=holding_period,
            direction=direction,
            benchmark_loader=benchmark_loader,
            report_writer=report_writer,
            engine_type=engine_type,
            dialect=dialect,
            canonical_expression=canonical_expression,
            canonical_ast=canonical_ast,
            diagnostics=diagnostics,
            start_date=start_date,
            end_date=end_date,
            metrics_source=metrics_source,
            execution_meta=execution_meta,
        )

    def compute_component_scores(
        self,
        *,
        rank_ic_mean: float,
        ic_ir: float,
        ic_win_rate: float,
        long_short_sharpe: float,
        long_short_annual: float,
        turnover: float,
        wq_fitness: float,
        max_drawdown: float,
    ) -> dict[str, Any]:
        ic_mean_score = min(abs(rank_ic_mean) / 0.05, 1.0) * 100
        ic_ir_score = min(abs(ic_ir) / 1.0, 1.0) * 100
        stability_score = min(max(ic_win_rate - 0.45, 0.0) / 0.25, 1.0) * 100
        sharpe_score = min(max(long_short_sharpe, 0.0) / 1.5, 1.0) * 100
        return_score = min(max(long_short_annual, 0.0) / 0.25, 1.0) * 100
        wq_alignment = min(max(wq_fitness, 0.0) / 1.5, 1.0) * 100
        turnover_score = max(0.0, 100.0 - min(turnover, 1.0) * 100.0)
        drawdown_score = max(0.0, 100.0 - min(max_drawdown, 1.0) * 100.0)
        total = (
            ic_mean_score * 0.2
            + ic_ir_score * 0.15
            + stability_score * 0.1
            + sharpe_score * 0.2
            + return_score * 0.15
            + wq_alignment * 0.12
            + turnover_score * 0.04
            + drawdown_score * 0.04
        )
        return {
            "ic_mean": round(ic_mean_score, 2),
            "ic_ir": round(ic_ir_score, 2),
            "stability": round(stability_score, 2),
            "group_backtest": round((sharpe_score + return_score) / 2, 2),
            "wq_alignment": round(wq_alignment, 2),
            "turnover": round(turnover_score, 2),
            "drawdown": round(drawdown_score, 2),
            "total_score": round(total, 2),
        }

    def build_anti_overfit_result(
        self,
        *,
        rank_ic_series: pd.Series,
        spread_series: pd.Series,
        turnover: float,
        max_drawdown: float,
    ) -> dict[str, Any]:
        tests: list[dict[str, Any]] = []
        passed = 0

        ic_stability = _safe_float(rank_ic_series.std())
        ic_pass = ic_stability <= 0.12
        if ic_pass:
            passed += 1
        tests.append({"name": "IC 稳定性", "passed": ic_pass, "details": f"rankIC 标准差 {ic_stability:.4f}，阈值 0.1200。"})

        spread_positive_rate = _safe_float((spread_series > 0).mean())
        spread_pass = spread_positive_rate >= 0.45
        if spread_pass:
            passed += 1
        tests.append({"name": "收益一致性", "passed": spread_pass, "details": f"Top-Bottom spread 为正的占比 {spread_positive_rate:.2%}，阈值 45%。"})

        turnover_pass = turnover <= 0.65
        if turnover_pass:
            passed += 1
        tests.append({"name": "换手约束", "passed": turnover_pass, "details": f"估算换手率 {turnover:.4f}，阈值 0.6500。"})

        drawdown_pass = max_drawdown <= 0.35
        if drawdown_pass:
            passed += 1
        tests.append({"name": "回撤约束", "passed": drawdown_pass, "details": f"最大回撤 {max_drawdown:.4f}，阈值 0.3500。"})

        score = round(passed / len(tests) * 100, 2) if tests else 0.0
        recommendation = "推荐" if passed >= 3 else "谨慎" if passed >= 2 else "需改进"
        return {"score": score, "recommendation": recommendation, "tests": tests}

    def build_interpretation(
        self,
        *,
        prompt: str,
        report_metrics: dict[str, Any],
        backtest_summary: dict[str, Any],
        wq_brain: dict[str, Any],
        anti_overfit: dict[str, Any],
    ) -> dict[str, Any]:
        weaknesses: list[str] = []
        ideas: list[str] = []

        if _safe_float(backtest_summary.get("rank_ic_mean")) < 0.02:
            weaknesses.append("横截面 rankIC 偏弱，说明选股区分度不足。")
            ideas.append("引入更强的排序或量价交互项，提升横截面区分度。")
        if _safe_float(report_metrics.get("sharpe")) < 0.6:
            weaknesses.append("L/S Sharpe 偏低，收益质量仍需改善。")
            ideas.append("优先降低噪音与波动暴露，减少极端反转信号。")
        if _safe_float(backtest_summary.get("turnover")) > 0.5:
            weaknesses.append("换手率偏高，可能影响真实可交易性。")
            ideas.append("通过更平滑的基础因子或中周期信号压低换手。")
        if _safe_float(report_metrics.get("max_drawdown")) > 0.25:
            weaknesses.append("最大回撤偏高，风险暴露需要约束。")
            ideas.append("补充防御性或波动率约束因子，改善回撤控制。")

        if not weaknesses:
            weaknesses.append("整体指标较均衡，但仍可围绕目标继续精修。")
        if not ideas:
            ideas.append("继续在当前因子族附近做结构性微调，优先保留高 rankIC 结构。")

        summary = (
            f"围绕“{prompt}”完成候选评估；当前评分 {wq_brain.get('wq_rating')}，"
            f"Sharpe {report_metrics.get('sharpe', 0):.2f}，WQ Fitness {backtest_summary.get('wq_fitness', 0):.2f}。"
        )
        return {
            "summary": summary,
            "weaknesses": weaknesses,
            "next_steps": ideas,
            "rating": wq_brain.get("wq_rating"),
            "rating_reason": anti_overfit.get("recommendation"),
            "improvement_ideas": ideas,
        }
