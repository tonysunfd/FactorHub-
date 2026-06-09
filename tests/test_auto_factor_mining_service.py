from __future__ import annotations

import asyncio
import pandas as pd
import pytest
from types import SimpleNamespace

from backend.services.auto_factor_mining_service import AutoFactorMiningService
from backend.services.factor_service import factor_service
from backend.services.research_tools.factor_selection_service import factor_selection_service
from backend.services.research_tools.validation_service import ValidationService


def test_build_continuation_context_uses_round_evaluation() -> None:
    service = AutoFactorMiningService()

    result = {
        "round_evaluation": {
            "base_factors": ["Alpha1", "Alpha2"],
            "primary_problem": "Sharpe 偏低",
            "recommended_goal": "ls_sharpe",
            "suggested_actions": ["补充波动率因子"],
        },
        "factors": [
            {
                "score": 72.5,
                "report_metrics": {"sharpe": 0.42, "max_drawdown": 0.28},
                "backtest_summary": {
                    "long_short_sharpe": 0.42,
                    "long_short_annual": 0.08,
                    "rank_ic_mean": 0.019,
                    "turnover": 0.61,
                    "wq_fitness": 0.78,
                },
                "interpretation": {
                    "weaknesses": ["横截面区分度不足"],
                    "next_steps": ["降低换手率"],
                },
                "expression": "rank(ts_mean(close/open, 5))",
                "raw_expression": "rank(ts_mean(close/open, 5))",
                "canonical_ast": {"fields": ["close", "open"], "operators": ["rank", "ts_mean"]},
            }
        ],
    }

    context = service.build_continuation_context(
        result=result,
        request_payload={"base_factors": ["Fallback"]},
        prompt="继续优化",
        factor_update_mode="append",
        additional_factor_count=4,
    )

    assert context["base_factors"] == ["Alpha1", "Alpha2"]
    assert context["primary_problem"] == "Sharpe 偏低"
    assert context["secondary_problem"] == "横截面区分度不足"
    assert context["recommended_goal"] == "ls_sharpe"
    assert "补充波动率因子" in context["suggested_actions"]
    assert context["parent_expression"] == "rank(ts_mean(close/open, 5))"
    assert context["parent_raw_expression"] == "rank(ts_mean(close/open, 5))"


def test_build_continuation_context_keeps_exploring_when_signal_is_only_moderately_good() -> None:
    service = AutoFactorMiningService()

    result = {
        "round_evaluation": {
            "base_factors": ["Alpha1", "Alpha2"],
            "primary_problem": "整体指标较均衡，但仍可围绕目标继续精修。",
            "recommended_goal": "ls_sharpe",
            "suggested_actions": ["继续在当前因子族附近做结构性微调，优先保留高 rankIC 结构。"],
        },
        "factors": [
            {
                "score": 68.03,
                "report_metrics": {"sharpe": 1.58, "max_drawdown": -0.14},
                "backtest_summary": {
                    "long_short_sharpe": 1.79,
                    "long_short_annual": 0.26,
                    "rank_ic_mean": 0.026,
                    "turnover": 0.49,
                    "wq_fitness": 1.30,
                },
                "interpretation": {
                    "weaknesses": ["整体指标较均衡，但仍可围绕目标继续精修。"],
                    "next_steps": ["继续在当前因子族附近做结构性微调，优先保留高 rankIC 结构。"],
                },
                "expression": "rank(ts_mean(close/open,5) * ts_mean((high-low)/open,5))",
                "raw_expression": "rank(ts_mean(close/open,5) * ts_mean((high-low)/open,5))",
            }
        ],
    }

    context = service.build_continuation_context(
        result=result,
        request_payload={"base_factors": ["Fallback"]},
        prompt="继续优化",
        factor_update_mode="append",
        additional_factor_count=2,
    )

    assert context["should_adjust_base_factors"] is True
    assert context["selection_confidence"] >= 2
    assert context["hold_reason"] == ""


def test_build_continuation_context_holds_when_metrics_are_already_strong() -> None:
    service = AutoFactorMiningService()

    result = {
        "round_evaluation": {
            "base_factors": ["Alpha1", "Alpha2"],
            "primary_problem": "Sharpe 可继续优化，但当前结果已经较稳定。",
            "recommended_goal": "ls_sharpe",
            "suggested_actions": ["围绕现有结构继续小步精修。"],
            "metric_snapshot": {
                "score": 82.44,
                "turnover": 0.36,
                "rank_ic": 0.0323,
                "ls_sharpe": 1.68,
            },
        },
        "factors": [
            {
                "score": 82.44,
                "report_metrics": {"sharpe": 1.68, "max_drawdown": -0.12, "cagr": 0.24},
                "backtest_summary": {
                    "long_short_sharpe": 1.68,
                    "long_short_annual": 0.24,
                    "rank_ic_mean": 0.0323,
                    "turnover": 0.36,
                    "wq_fitness": 1.22,
                },
                "interpretation": {
                    "weaknesses": ["Sharpe 可继续优化，但当前结果已经较稳定。"],
                    "next_steps": ["围绕现有结构继续小步精修。"],
                },
            }
        ],
    }

    context = service.build_continuation_context(
        result=result,
        request_payload={"base_factors": ["Alpha1", "Alpha2"]},
        prompt="继续优化",
        factor_update_mode="append",
        additional_factor_count=2,
    )

    assert context["should_adjust_base_factors"] is False
    assert "保持当前基础因子组合" in context["hold_reason"]


def test_build_continuation_context_replaces_unused_new_factors_even_when_metrics_are_good() -> None:
    service = AutoFactorMiningService()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        service,
        "resolve_base_factor_codes",
        lambda base_factors: ["close/open", "(high-low)/open", "force_index_ma", "ts_std(log(close/ts_shift(close,1)),10)"],
    )

    result = {
        "round_evaluation": {
            "base_factors": ["Alpha1", "Alpha2", "ForceIndex", "Volatility10"],
            "primary_problem": "Sharpe 可继续优化，但当前结果已经较稳定。",
            "recommended_goal": "ls_sharpe",
            "suggested_actions": ["围绕现有结构继续小步精修。"],
            "metric_snapshot": {
                "score": 81.14,
                "turnover": 0.505,
                "rank_ic": 0.0717,
                "ls_sharpe": 3.179,
            },
        },
        "factors": [
            {
                "score": 81.14,
                "report_metrics": {"sharpe": 2.97, "max_drawdown": -0.08, "cagr": 0.61},
                "backtest_summary": {
                    "long_short_sharpe": 3.179,
                    "long_short_annual": 0.49,
                    "rank_ic_mean": 0.0717,
                    "turnover": 0.505,
                    "wq_fitness": 3.14,
                },
                "interpretation": {
                    "weaknesses": ["Sharpe 可继续优化，但当前结果已经较稳定。"],
                    "next_steps": ["围绕现有结构继续小步精修。"],
                },
                "expression": "rank(ts_mean(close/open,5) * ts_mean(force_index_ma,5))",
                "raw_expression": "rank(ts_mean(close/open,5) * ts_mean(force_index_ma,5))",
                "task_details": {
                    "round_evaluation": {
                        "parent_expression": "rank(ts_mean(close/open,5))",
                    }
                },
            }
        ],
    }

    context = service.build_continuation_context(
        result=result,
        request_payload={"base_factors": ["Alpha1", "Alpha2"]},
        prompt="继续优化",
        factor_update_mode="append",
        additional_factor_count=2,
    )

    monkeypatch.undo()
    assert context["should_adjust_base_factors"] is True
    assert context["replace_base_factors"] == ["Volatility10"]


def test_filter_retained_factors_supports_match_modes() -> None:
    service = AutoFactorMiningService()
    factors = [
        {
            "score": 82,
            "backtest_summary": {"long_short_sharpe": 1.1, "long_short_annual": 0.18},
            "wq_brain": {"wq_rating": "A", "wq_returns": 0.18},
        },
        {
            "score": 58,
            "backtest_summary": {"long_short_sharpe": 0.4, "long_short_annual": 0.03},
            "wq_brain": {"wq_rating": "C", "wq_returns": 0.03},
        },
    ]

    retained_all = service.filter_retained_factors(
        factors,
        {"match_mode": "all", "score_min": 60, "wq_ratings": ["A", "B"], "ls_sharpe_min": 0.8},
    )
    retained_any = service.filter_retained_factors(
        factors,
        {"match_mode": "any", "score_min": 80, "wq_ratings": ["C"]},
    )

    assert retained_all == [factors[0]]
    assert retained_any == factors


def test_filter_retained_factors_falls_back_to_best_ranked_candidates_when_thresholds_miss() -> None:
    service = AutoFactorMiningService()
    factors = [
        {
            "name": "BestNearMiss",
            "score": 79,
            "backtest_summary": {"long_short_sharpe": 0.95, "long_short_annual": 0.17, "rank_ic_mean": 0.028, "turnover": 0.32},
            "wq_brain": {"wq_rating": "B", "wq_returns": 0.15, "wq_fitness": 1.1},
        },
        {
            "name": "WeakerNearMiss",
            "score": 70,
            "backtest_summary": {"long_short_sharpe": 0.75, "long_short_annual": 0.09, "rank_ic_mean": 0.012, "turnover": 0.58},
            "wq_brain": {"wq_rating": "C", "wq_returns": 0.08, "wq_fitness": 0.6},
        },
    ]

    retained = service.filter_retained_factors(
        factors,
        {"match_mode": "all", "score_min": 85, "ls_sharpe_min": 1.2, "ls_return_min": 0.2},
    )

    assert [item["name"] for item in retained] == ["BestNearMiss", "WeakerNearMiss"]


def test_run_auto_campaign_returns_real_rounds(monkeypatch) -> None:
    service = AutoFactorMiningService()
    continuation_calls: list[dict[str, object]] = []
    prompts_seen: list[str] = []

    def fake_run_auto_mining(**kwargs):
        prompts_seen.append(kwargs["prompt"])
        base_factors = kwargs["base_factors"]
        round_index = 1 if "ExtraFactor" not in base_factors else 2
        best_score = 60 + round_index * 10
        expression = "rank(ts_mean(close/open,5))" if round_index == 1 else "rank(ts_mean(close/open,5) * ts_mean(volume,5))"
        return {
            "factors": [
                {
                    "name": f"Auto_{round_index}",
                    "expression": expression,
                    "score": best_score,
                    "grade": "B" if round_index == 1 else "A",
                    "report_metrics": {"sharpe": 0.5 + round_index * 0.2, "cagr": 0.05 + round_index * 0.03},
                    "backtest_summary": {
                        "long_short_sharpe": 0.5 + round_index * 0.2,
                        "long_short_annual": 0.05 + round_index * 0.03,
                        "rank_ic_mean": 0.02 + round_index * 0.01,
                        "ic_ir": 0.6 + round_index * 0.1,
                        "turnover": 0.3,
                        "wq_fitness": 0.8 + round_index * 0.1,
                    },
                    "wq_brain": {
                        "wq_rating": "B" if round_index == 1 else "A",
                        "wq_returns": 0.05 + round_index * 0.03,
                        "wq_fitness": 0.8 + round_index * 0.1,
                    },
                    "interpretation": {
                        "weaknesses": [f"第 {round_index} 轮短板"],
                        "next_steps": [f"第 {round_index} 轮建议"],
                    },
                    "task_details": {
                        "round_evaluation": {
                            "base_factors": list(base_factors),
                            "primary_problem": f"第 {round_index} 轮问题",
                            "recommended_goal": "score",
                            "suggested_actions": [f"第 {round_index} 轮动作"],
                            "metric_snapshot": {"score": best_score},
                        }
                    },
                }
            ],
            "best_score": best_score,
            "avg_score": best_score - 5,
            "generations": 1,
            "fitness_history": {"best": [best_score], "average": [best_score - 5]},
            "round_evaluation": {
                "base_factors": list(base_factors),
                "primary_problem": f"第 {round_index} 轮问题",
                "recommended_goal": "score",
                "suggested_actions": [f"第 {round_index} 轮动作"],
                "metric_snapshot": {"score": best_score},
            },
        }

    def fake_select_continue_factors(**kwargs):
        continuation_calls.append(kwargs)
        return {
            "selected_factors": ["ExtraFactor"],
            "selection_rationale": "补充一个新因子",
            "per_factor_reason": {"ExtraFactor": "用于下一轮探索"},
            "continuation_context": {
                "primary_problem": "第 1 轮问题",
                "recommended_goal": "score",
                "suggested_actions": ["第 1 轮动作"],
                "summary_text": "沿着第 1 轮问题继续优化",
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)
    monkeypatch.setattr(
        service,
        "resolve_base_factor_codes",
        lambda base_factors: ["close/open" if item == "Alpha1" else "volume" for item in base_factors],
    )

    snapshots = []
    result = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
        progress_callback=snapshots.append,
    )

    assert len(result["rounds"]) == 2
    assert result["rounds"][1]["input_base_factors"] == ["Alpha1", "ExtraFactor"]
    assert result["best_score"] == 80
    assert result["retained_factors"][0]["expression"] == "rank(ts_mean(close/open,5) * ts_mean(volume,5))"
    assert result["latest_round_retained_factors"][0]["expression"] == "rank(ts_mean(close/open,5) * ts_mean(volume,5))"
    assert result["final_round_result"]["best_score"] == 80
    assert result["latest_round_result"]["best_score"] == 80
    assert result["best_parent_result"]["best_score"] == 80
    assert snapshots[-1]["current_round"] == 2
    assert continuation_calls[0]["parent_result"]["round_evaluation"]["primary_problem"] == "第 1 轮问题"
    assert result["rounds"][1]["continuation_hypothesis"]["reason"] == "第 1 轮问题"
    assert result["rounds"][1]["continuation_hypothesis"]["target_goal"] == "score"
    assert result["rounds"][1]["continuation_hypothesis"]["candidate_factors"] == ["ExtraFactor"]
    assert result["rounds"][1]["continuation_hypothesis"]["selected_for_next_round"] == ["ExtraFactor"]
    assert result["rounds"][1]["factor_usage"]["used_base_factors"] == ["Alpha1", "ExtraFactor"]
    assert result["rounds"][1]["factor_usage"]["used_new_factors"] == ["ExtraFactor"]
    assert result["rounds"][1]["factor_usage"]["unused_new_factors"] == []
    assert prompts_seen[0] == "提升综合分数"
    assert "连续探索第 2 轮补充要求" in prompts_seen[1]
    assert "本轮唯一主目标是 score" in prompts_seen[1]
    assert "优先动作：" in prompts_seen[1]
    assert "第 1 轮动作" in prompts_seen[1]


def test_run_auto_campaign_progress_uses_current_round_snapshot(monkeypatch) -> None:
    service = AutoFactorMiningService()

    def fake_run_auto_mining(**kwargs):
        base_factors = kwargs["base_factors"]
        progress_callback = kwargs["progress_callback"]
        round_index = 1 if "ExtraFactor" not in base_factors else 2
        candidate = {
            "name": f"Auto_{round_index}",
            "expression": "rank(ts_mean(close/open,5))" if round_index == 1 else "rank(ts_mean(close/open,5) * ts_mean(volume,5))",
            "score": 60 + round_index * 10,
        }
        progress_callback(1, 2, candidate)
        return {
            "factors": [candidate],
            "best_score": candidate["score"],
            "avg_score": candidate["score"] - 5,
            "generations": 1,
            "fitness_history": {"best": [candidate["score"]], "average": [candidate["score"] - 5]},
            "round_evaluation": {
                "base_factors": list(base_factors),
                "primary_problem": f"第 {round_index} 轮问题",
                "recommended_goal": "score",
                "suggested_actions": [f"第 {round_index} 轮动作"],
                "metric_snapshot": {"score": candidate["score"]},
            },
        }

    def fake_select_continue_factors(**kwargs):
        return {
            "selected_factors": ["ExtraFactor"],
            "selection_rationale": "补充一个新因子",
            "per_factor_reason": {"ExtraFactor": "用于下一轮探索"},
            "continuation_context": {
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)
    monkeypatch.setattr(
        service,
        "resolve_base_factor_codes",
        lambda base_factors: ["close/open" if item == "Alpha1" else "volume" for item in base_factors],
    )

    snapshots = []
    service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=2,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
        progress_callback=snapshots.append,
    )

    in_progress_round_two = next(
        snapshot
        for snapshot in snapshots
        if snapshot["current_round"] == 2 and snapshot["current_generation"] == 1
    )
    latest_round = in_progress_round_two["latest_round"]

    assert latest_round["round_index"] == 2
    assert latest_round["continuation_feedback"] is None
    assert latest_round["input_base_factors"] == ["Alpha1", "ExtraFactor"]
    assert latest_round["all_factors"][0]["expression"] == "rank(ts_mean(close/open,5) * ts_mean(volume,5))"
    assert latest_round["factor_usage"]["used_new_factors"] == ["ExtraFactor"]


def test_run_auto_campaign_progress_caps_displayed_generation_count(monkeypatch) -> None:
    service = AutoFactorMiningService()

    def fake_run_auto_mining(**kwargs):
        progress_callback = kwargs["progress_callback"]
        for idx in range(3):
            progress_callback(
                idx + 1,
                2,
                {
                    "name": f"Auto_{idx + 1}",
                    "expression": f"expr_{idx + 1}",
                    "score": 60 + idx,
                },
            )
        return {
            "factors": [{"name": "Auto_final", "expression": "expr_final", "score": 88}],
            "best_score": 88,
            "avg_score": 70,
            "generations": 1,
            "fitness_history": {"best": [88], "average": [70]},
            "round_evaluation": {
                "base_factors": ["Alpha1"],
                "primary_problem": "第 1 轮问题",
                "recommended_goal": "score",
                "suggested_actions": ["第 1 轮动作"],
                "metric_snapshot": {"score": 88},
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(
        service,
        "resolve_base_factor_codes",
        lambda base_factors: ["close/open" for _ in base_factors],
    )

    snapshots = []
    service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=1,
        n_candidates_per_round=2,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
        progress_callback=snapshots.append,
    )

    assert any(snapshot["current_generation"] == 2 for snapshot in snapshots)
    assert all(snapshot["current_generation"] <= snapshot["total_generations"] for snapshot in snapshots)


def test_build_factor_usage_summary_recognizes_adapted_new_factor_code(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        service,
        "resolve_base_factor_codes",
        lambda base_factors: [
            "close / open",
            "close / (SUM(close * volume, 20) / SUM(volume, 20))",
        ],
    )

    summary = service._build_factor_usage_summary(
        current_base_factors=["close_open_ratio", "price_vwma_ratio"],
        previous_base_factors=["close_open_ratio"],
        best_factor={
            "expression": "ts_mean(ts_rank(close/open,15),3)*ts_mean(ts_rank(close/(ts_sum(close*volume,20)/ts_sum(volume,20)),20),5)",
            "canonical_expression": "ts_mean(ts_rank(close/open,15),3)*ts_mean(ts_rank(close/(ts_sum(close*volume,20)/ts_sum(volume,20)),20),5)",
        },
    )

    assert summary["used_base_factors"] == ["close_open_ratio", "price_vwma_ratio"]
    assert summary["unused_base_factors"] == []
    assert summary["used_new_factors"] == ["price_vwma_ratio"]
    assert summary["unused_new_factors"] == []


def test_build_factor_usage_summary_treats_parent_seed_continuation_as_new_factor_usage(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        service,
        "resolve_base_factor_codes",
        lambda base_factors: [
            "close/open",
            "(high-low)/open",
            "ts_mean((close-ts_shift(close,1))*volume,timeperiod=13)",
            "ts_std(log(close/ts_shift(close,1)),10)",
        ],
    )

    parent_expression = "rank(ts_mean(ts_rank(sma((close/open)-1,5),20)*ts_rank(sma(volume/sma(volume,20),5),20)-ts_rank(ts_std((close/open)-1,10),20),5)*ts_mean(ts_mean((close-ts_shift(close,1))*volume,timeperiod=13),5))"
    summary = service._build_factor_usage_summary(
        current_base_factors=["close_open_ratio", "high_low_ratio", "force_index_ma", "volatility_10"],
        previous_base_factors=["close_open_ratio", "high_low_ratio"],
        best_factor={
            "expression": "rank(ts_mean((ts_rank(sma((close/open)-1,5),20)*ts_rank(sma(volume/sma(volume,20),5),20)-ts_rank(ts_std(log(close/ts_shift(close,1)),10),20)),5)*ts_mean(ts_mean((close-ts_shift(close,1))*volume,timeperiod=13),5)/(1+ts_mean((high-low)/open,5)))",
            "task_details": {
                "round_evaluation": {
                    "parent_expression": parent_expression,
                }
            },
        },
    )

    assert "force_index_ma" in summary["used_base_factors"]
    assert "force_index_ma" in summary["used_new_factors"]
    assert "volatility_10" in summary["used_new_factors"]


def test_run_auto_campaign_round_task_ids_are_unique_across_campaigns(monkeypatch) -> None:
    service = AutoFactorMiningService()

    def fake_run_auto_mining(**kwargs):
        base_factors = list(kwargs["base_factors"])
        expression = "_".join(base_factors)
        return {
            "factors": [
                {
                    "name": f"Auto_{expression}",
                    "expression": f"expr_{expression}",
                    "score": 70.0,
                    "grade": "A",
                    "report_metrics": {"sharpe": 1.0},
                    "backtest_summary": {
                        "long_short_sharpe": 1.0,
                        "long_short_annual": 0.1,
                        "rank_ic_mean": 0.02,
                        "turnover": 0.3,
                    },
                }
            ],
            "best_score": 70.0,
            "avg_score": 68.0,
            "generations": 1,
            "fitness_history": {"best": [70.0], "average": [68.0]},
            "round_evaluation": {
                "base_factors": base_factors,
                "primary_problem": "第 1 轮问题",
                "recommended_goal": "score",
                "suggested_actions": ["继续优化"],
                "metric_snapshot": {"score": 70.0},
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(
        service,
        "select_continue_factors",
        lambda **kwargs: {
            "selected_factors": ["ExtraFactor"],
            "selection_rationale": "补一个新因子",
            "per_factor_reason": {"ExtraFactor": "用于下一轮探索"},
            "continuation_context": {
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
            },
        },
    )

    result_one = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
    )
    result_two = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
    )

    round_ids_one = [round_item["task_id"] for round_item in result_one["rounds"]]
    round_ids_two = [round_item["task_id"] for round_item in result_two["rounds"]]
    assert len(set(round_ids_one)) == len(round_ids_one)
    assert len(set(round_ids_two)) == len(round_ids_two)
    assert set(round_ids_one).isdisjoint(set(round_ids_two))


def test_run_auto_campaign_keeps_base_factors_when_signal_is_weak(monkeypatch) -> None:
    service = AutoFactorMiningService()

    def fake_run_auto_mining(**kwargs):
        base_factors = list(kwargs["base_factors"])
        round_index = 1 if base_factors == ["Alpha1"] else 2
        best_score = 68.0 if round_index == 1 else 66.0
        return {
            "factors": [
                {
                    "name": f"Auto_{round_index}",
                    "expression": f"expr_{round_index}",
                    "score": best_score,
                    "grade": "B",
                    "report_metrics": {"sharpe": 1.5, "cagr": 0.2},
                    "backtest_summary": {
                        "long_short_sharpe": 1.7,
                        "long_short_annual": 0.22,
                        "rank_ic_mean": 0.026,
                        "ic_ir": 0.13,
                        "turnover": 0.48,
                        "wq_fitness": 1.2,
                    },
                    "wq_brain": {
                        "wq_rating": "B",
                        "wq_returns": 0.22,
                        "wq_fitness": 1.2,
                    },
                    "interpretation": {
                        "weaknesses": ["整体指标较均衡，但仍可围绕目标继续精修。"] if round_index == 1 else ["第 2 轮短板"],
                        "next_steps": ["继续在当前因子族附近做结构性微调，优先保留高 rankIC 结构。"] if round_index == 1 else ["第 2 轮建议"],
                    },
                    "task_details": {
                        "round_evaluation": {
                            "base_factors": base_factors,
                            "primary_problem": "整体指标较均衡，但仍可围绕目标继续精修。" if round_index == 1 else "第 2 轮问题",
                            "recommended_goal": "ls_sharpe",
                            "suggested_actions": ["继续在当前因子族附近做结构性微调，优先保留高 rankIC 结构。"] if round_index == 1 else ["第 2 轮动作"],
                            "metric_snapshot": {"score": best_score, "turnover": 0.48, "rank_ic": 0.026, "ls_sharpe": 1.7},
                        }
                    },
                }
            ],
            "best_score": best_score,
            "avg_score": best_score,
            "generations": 1,
            "fitness_history": {"best": [best_score], "average": [best_score]},
            "round_evaluation": {
                "base_factors": base_factors,
                "primary_problem": "整体指标较均衡，但仍可围绕目标继续精修。" if round_index == 1 else "第 2 轮问题",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["继续在当前因子族附近做结构性微调，优先保留高 rankIC 结构。"] if round_index == 1 else ["第 2 轮动作"],
                "metric_snapshot": {"score": best_score, "turnover": 0.48, "rank_ic": 0.026, "ls_sharpe": 1.7},
            },
        }

    def fake_select_continue_factors(**kwargs):
        return {
            "selected_factors": ["ExtraFactor"],
            "selection_rationale": "候选可供参考",
            "per_factor_reason": {"ExtraFactor": "可尝试"},
            "continuation_context": {
                "primary_problem": "整体指标较均衡，但仍可围绕目标继续精修。",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["继续在当前因子族附近做结构性微调，优先保留高 rankIC 结构。"],
                "summary_text": "沿着当前因子族继续优化",
                "should_adjust_base_factors": False,
                "hold_reason": "上一轮没有暴露出足够明确的结构性短板，优先保持当前基础因子组合，先在表达式结构上做微调。",
                "selection_confidence": 0,
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)

    result = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
    )

    assert result["rounds"][1]["input_base_factors"] == ["Alpha1"]
    assert result["rounds"][1]["continuation_hypothesis"]["candidate_factors"] == ["ExtraFactor"]
    assert result["rounds"][1]["continuation_hypothesis"]["selected_for_next_round"] == []
    assert result["rounds"][1]["continuation_hypothesis"]["should_adjust_base_factors"] is False
    assert "保持当前基础因子组合" in result["rounds"][1]["continuation_hypothesis"]["hold_reason"]
    assert result["rounds"][0]["continuation_hypothesis"] is None


def test_run_auto_campaign_best_score_parent_strategy_reuses_best_round_as_next_parent(monkeypatch) -> None:
    service = AutoFactorMiningService()
    continuation_calls: list[dict[str, object]] = []
    selection_call_count = {"count": 0}

    def fake_run_auto_mining(**kwargs):
        base_factors = list(kwargs["base_factors"])
        if base_factors == ["Alpha1"]:
            round_index = 1
            best_score = 80.0
            primary_problem = "第 1 轮问题"
        elif base_factors == ["BetaFactor"]:
            round_index = 2
            best_score = 72.0
            primary_problem = "第 2 轮问题"
        elif base_factors == ["DeltaFactor"]:
            round_index = 3
            best_score = 78.0
            primary_problem = "第 3 轮问题"
        else:
            raise AssertionError(f"unexpected base factors: {base_factors}")

        return {
            "factors": [
                {
                    "name": f"Auto_{round_index}",
                    "expression": f"expr_{round_index}",
                    "score": best_score,
                    "grade": "A",
                    "report_metrics": {"sharpe": 1.0},
                    "backtest_summary": {
                        "long_short_sharpe": 1.0,
                        "long_short_annual": 0.1,
                        "rank_ic_mean": 0.02,
                        "ic_ir": 0.5,
                        "turnover": 0.3,
                        "wq_fitness": 1.0,
                    },
                    "wq_brain": {"wq_rating": "A", "wq_returns": 0.1, "wq_fitness": 1.0},
                    "interpretation": {"weaknesses": [primary_problem], "next_steps": [f"第 {round_index} 轮动作"]},
                    "task_details": {
                        "round_evaluation": {
                            "base_factors": base_factors,
                            "primary_problem": primary_problem,
                            "recommended_goal": "score",
                            "suggested_actions": [f"第 {round_index} 轮动作"],
                            "metric_snapshot": {"score": best_score},
                        }
                    },
                }
            ],
            "best_score": best_score,
            "avg_score": best_score - 2,
            "generations": 1,
            "fitness_history": {"best": [best_score], "average": [best_score - 2]},
            "round_evaluation": {
                "base_factors": base_factors,
                "primary_problem": primary_problem,
                "recommended_goal": "score",
                "suggested_actions": [f"第 {round_index} 轮动作"],
                "metric_snapshot": {"score": best_score},
            },
        }

    def fake_select_continue_factors(**kwargs):
        continuation_calls.append(kwargs)
        selection_call_count["count"] += 1
        parent_base_factors = list((kwargs.get("parent_request") or {}).get("base_factors") or [])
        if selection_call_count["count"] == 1:
            assert parent_base_factors == ["Alpha1"]
            return {
                "selected_factors": ["BetaFactor"],
                "selection_rationale": "第 1 轮后补一个新因子",
                "per_factor_reason": {"BetaFactor": "第 1 轮建议"},
                "continuation_context": {
                    "primary_problem": "第 1 轮问题",
                    "recommended_goal": "score",
                    "suggested_actions": ["第 1 轮动作"],
                    "summary_text": "沿着第 1 轮问题继续优化",
                    "should_adjust_base_factors": True,
                    "selection_confidence": 3,
                },
            }
        assert parent_base_factors == ["Alpha1"]
        return {
            "selected_factors": ["DeltaFactor"],
            "selection_rationale": "继续复用第 1 轮 parent",
            "per_factor_reason": {"DeltaFactor": "基于最佳轮继续探索"},
            "continuation_context": {
                "primary_problem": "第 1 轮问题",
                "recommended_goal": "score",
                "suggested_actions": ["第 1 轮动作"],
                "summary_text": "继续沿着第 1 轮问题优化",
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)

    result = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=3,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="reselect",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
    )

    assert len(continuation_calls) == 2
    assert continuation_calls[1]["parent_result"]["round_evaluation"]["primary_problem"] == "第 1 轮问题"
    assert continuation_calls[1]["parent_request"]["base_factors"] == ["Alpha1"]
    assert result["rounds"][2]["input_base_factors"] == ["DeltaFactor"]
    assert result["rounds"][2]["previous_base_factors"] == ["BetaFactor"]
    assert result["rounds"][2]["continuation_hypothesis"]["reason"] == "第 2 轮问题"
    assert result["rounds"][1]["continuation_hypothesis"]["next_round_parent_base_factors"] == ["Alpha1"]


def test_run_auto_campaign_append_mode_uses_best_parent_base_factors_for_next_round(monkeypatch) -> None:
    service = AutoFactorMiningService()
    selection_call_count = {"count": 0}

    def fake_run_auto_mining(**kwargs):
        base_factors = list(kwargs["base_factors"])
        if base_factors == ["Alpha1"]:
            round_index = 1
            best_score = 90.0
            primary_problem = "第 1 轮问题"
        elif base_factors == ["Alpha1", "BetaFactor"]:
            round_index = 2
            best_score = 70.0
            primary_problem = "第 2 轮问题"
        elif base_factors == ["Alpha1", "GammaFactor"]:
            round_index = 3
            best_score = 75.0
            primary_problem = "第 3 轮问题"
        else:
            raise AssertionError(f"unexpected base factors: {base_factors}")

        return {
            "factors": [
                {
                    "name": f"Auto_{round_index}",
                    "expression": f"expr_{round_index}",
                    "score": best_score,
                    "grade": "A",
                    "report_metrics": {"sharpe": 1.0},
                    "backtest_summary": {
                        "long_short_sharpe": 1.0,
                        "long_short_annual": 0.1,
                        "rank_ic_mean": 0.02,
                        "ic_ir": 0.5,
                        "turnover": 0.3,
                        "wq_fitness": 1.0,
                    },
                    "wq_brain": {"wq_rating": "A", "wq_returns": 0.1, "wq_fitness": 1.0},
                    "interpretation": {"weaknesses": [primary_problem], "next_steps": [f"第 {round_index} 轮动作"]},
                    "task_details": {
                        "round_evaluation": {
                            "base_factors": base_factors,
                            "primary_problem": primary_problem,
                            "recommended_goal": "score",
                            "suggested_actions": [f"第 {round_index} 轮动作"],
                            "metric_snapshot": {"score": best_score},
                        }
                    },
                }
            ],
            "best_score": best_score,
            "avg_score": best_score - 2,
            "generations": 1,
            "fitness_history": {"best": [best_score], "average": [best_score - 2]},
            "round_evaluation": {
                "base_factors": base_factors,
                "primary_problem": primary_problem,
                "recommended_goal": "score",
                "suggested_actions": [f"第 {round_index} 轮动作"],
                "metric_snapshot": {"score": best_score},
            },
        }

    def fake_select_continue_factors(**kwargs):
        selection_call_count["count"] += 1
        if selection_call_count["count"] == 1:
            return {
                "selected_factors": ["BetaFactor"],
                "selection_rationale": "补充第一个新因子",
                "per_factor_reason": {"BetaFactor": "第 1 轮建议"},
                "continuation_context": {
                    "primary_problem": "第 1 轮问题",
                    "recommended_goal": "score",
                    "suggested_actions": ["第 1 轮动作"],
                    "summary_text": "沿着第 1 轮问题继续优化",
                    "should_adjust_base_factors": True,
                    "selection_confidence": 3,
                },
            }
        assert list((kwargs.get("parent_request") or {}).get("base_factors") or []) == ["Alpha1"]
        return {
            "selected_factors": ["GammaFactor"],
            "selection_rationale": "基于最佳轮继续探索",
            "per_factor_reason": {"GammaFactor": "继续沿用第 1 轮 parent"},
            "continuation_context": {
                "primary_problem": "第 1 轮问题",
                "recommended_goal": "score",
                "suggested_actions": ["第 1 轮动作"],
                "summary_text": "继续沿着第 1 轮问题优化",
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)

    result = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=3,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
    )

    assert result["rounds"][1]["input_base_factors"] == ["Alpha1", "BetaFactor"]
    assert result["rounds"][2]["input_base_factors"] == ["Alpha1", "GammaFactor"]
    assert result["rounds"][2]["previous_base_factors"] == ["Alpha1", "BetaFactor"]
    assert result["rounds"][1]["continuation_hypothesis"]["next_round_parent_base_factors"] == ["Alpha1"]


def test_run_auto_campaign_append_mode_replaces_unused_new_factors(monkeypatch) -> None:
    service = AutoFactorMiningService()
    selection_call_count = {"count": 0}

    def fake_run_auto_mining(**kwargs):
        base_factors = list(kwargs["base_factors"])
        if base_factors == ["Alpha1", "BetaFactor"]:
            best_score = 80.0
            primary_problem = "第 2 轮问题"
            expression = "rank(ts_mean(close/open,5))"
        elif base_factors == ["Alpha1", "GammaFactor"]:
            best_score = 82.0
            primary_problem = "第 3 轮问题"
            expression = "rank(ts_mean(close/open,5) * ts_mean(volume,5))"
        else:
            best_score = 70.0
            primary_problem = "第 1 轮问题"
            expression = "rank(ts_mean(close/open,5))"
        return {
            "factors": [
                {
                    "name": "Auto",
                    "expression": expression,
                    "score": best_score,
                    "grade": "A",
                    "report_metrics": {"sharpe": 1.0},
                    "backtest_summary": {
                        "long_short_sharpe": 1.0,
                        "long_short_annual": 0.1,
                        "rank_ic_mean": 0.02,
                        "ic_ir": 0.5,
                        "turnover": 0.3,
                        "wq_fitness": 1.0,
                    },
                    "wq_brain": {"wq_rating": "A", "wq_returns": 0.1, "wq_fitness": 1.0},
                    "interpretation": {"weaknesses": [primary_problem], "next_steps": [primary_problem]},
                    "task_details": {
                        "round_evaluation": {
                            "base_factors": base_factors,
                            "primary_problem": primary_problem,
                            "recommended_goal": "score",
                            "suggested_actions": [primary_problem],
                            "metric_snapshot": {"score": best_score},
                        }
                    },
                }
            ],
            "best_score": best_score,
            "avg_score": best_score - 1,
            "generations": 1,
            "fitness_history": {"best": [best_score], "average": [best_score - 1]},
            "round_evaluation": {
                "base_factors": base_factors,
                "primary_problem": primary_problem,
                "recommended_goal": "score",
                "suggested_actions": [primary_problem],
                "metric_snapshot": {"score": best_score},
            },
        }

    def fake_select_continue_factors(**kwargs):
        selection_call_count["count"] += 1
        if selection_call_count["count"] == 1:
            return {
                "selected_factors": ["BetaFactor"],
                "selection_rationale": "补充第一个新因子",
                "per_factor_reason": {"BetaFactor": "第 1 轮建议"},
                "continuation_context": {
                    "primary_problem": "第 1 轮问题",
                    "recommended_goal": "score",
                    "suggested_actions": ["第 1 轮动作"],
                    "summary_text": "沿着第 1 轮问题继续优化",
                    "should_adjust_base_factors": True,
                    "selection_confidence": 3,
                    "replace_base_factors": [],
                },
            }
        return {
            "selected_factors": ["GammaFactor"],
            "selection_rationale": "替换未生效新因子",
            "per_factor_reason": {"GammaFactor": "替换未生效 BetaFactor"},
            "continuation_context": {
                "primary_problem": "第 2 轮问题",
                "recommended_goal": "score",
                "suggested_actions": ["第 2 轮动作"],
                "summary_text": "替换未生效新因子",
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
                "replace_base_factors": ["BetaFactor"],
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)

    result = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=3,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
    )

    assert result["rounds"][1]["input_base_factors"] == ["Alpha1", "BetaFactor"]
    assert result["rounds"][2]["input_base_factors"] == ["Alpha1", "GammaFactor"]
    assert result["rounds"][2]["continuation_hypothesis"]["replace_base_factors"] == ["BetaFactor"]


def test_continuation_feedback_rejects_score_gain_when_target_metrics_regress() -> None:
    service = AutoFactorMiningService()

    previous_best_result = {
        "best_score": 50.0,
        "factors": [
            {
                "score": 50.0,
                "report_metrics": {"sharpe": 1.2},
                "backtest_summary": {
                    "rank_ic_mean": 0.022,
                    "long_short_sharpe": 0.8,
                    "long_short_annual": 0.16,
                    "turnover": 0.32,
                },
            }
        ],
        "round_evaluation": {
            "metric_snapshot": {
                "score": 50.0,
                "rank_ic": 0.022,
                "ls_sharpe": 0.8,
                "ls_return": 0.16,
                "turnover": 0.32,
                "report_sharpe": 1.2,
            }
        },
    }
    current_result = {
        "best_score": 52.0,
        "factors": [
            {
                "score": 52.0,
                "report_metrics": {"sharpe": 0.95},
                "backtest_summary": {
                    "rank_ic_mean": 0.015,
                    "long_short_sharpe": 0.73,
                    "long_short_annual": 0.15,
                    "turnover": 0.45,
                },
            }
        ],
        "round_evaluation": {
            "metric_snapshot": {
                "score": 52.0,
                "rank_ic": 0.015,
                "ls_sharpe": 0.73,
                "ls_return": 0.15,
                "turnover": 0.45,
                "report_sharpe": 0.95,
            }
        },
    }

    feedback = service._build_continuation_feedback(
        previous_best_score=50.0,
        current_best_score=52.0,
        retention_count=1,
        direction="ls_sharpe",
        previous_best_result=previous_best_result,
        current_result=current_result,
    )

    assert feedback["decision"] is True
    assert feedback["accepted_as_best"] is False
    assert feedback["fallback_parent_strategy"] == "best_score_so_far"
    assert feedback["metric_deltas"]["ls_sharpe"] < 0
    assert feedback["metric_deltas"]["rank_ic"] < 0
    assert "退化" in feedback["hypothesis_evaluation"] or "回退" in feedback["reason"]


def test_continuation_feedback_accepts_target_metric_improvement_despite_small_score_drop() -> None:
    service = AutoFactorMiningService()

    previous_best_result = {
        "best_score": 60.0,
        "factors": [
            {
                "score": 60.0,
                "report_metrics": {"sharpe": 0.95},
                "backtest_summary": {
                    "rank_ic_mean": 0.022,
                    "long_short_sharpe": 0.78,
                    "long_short_annual": 0.16,
                    "turnover": 0.32,
                },
            }
        ],
        "round_evaluation": {
            "metric_snapshot": {
                "score": 60.0,
                "rank_ic": 0.022,
                "ls_sharpe": 0.78,
                "ls_return": 0.16,
                "turnover": 0.32,
                "report_sharpe": 0.95,
            }
        },
    }
    current_result = {
        "best_score": 59.4,
        "factors": [
            {
                "score": 59.4,
                "report_metrics": {"sharpe": 1.04},
                "backtest_summary": {
                    "rank_ic_mean": 0.0212,
                    "long_short_sharpe": 0.86,
                    "long_short_annual": 0.169,
                    "turnover": 0.34,
                },
            }
        ],
        "round_evaluation": {
            "metric_snapshot": {
                "score": 59.4,
                "rank_ic": 0.0212,
                "ls_sharpe": 0.86,
                "ls_return": 0.169,
                "turnover": 0.34,
                "report_sharpe": 1.04,
            }
        },
    }

    feedback = service._build_continuation_feedback(
        previous_best_score=60.0,
        current_best_score=59.4,
        retention_count=1,
        direction="ls_sharpe",
        previous_best_result=previous_best_result,
        current_result=current_result,
    )

    assert feedback["decision"] is True
    assert feedback["accepted_as_best"] is True
    assert feedback["fallback_parent_strategy"] is None
    assert feedback["metric_deltas"]["score"] < 0
    assert feedback["metric_deltas"]["ls_sharpe"] > 0
    assert "主目标指标已有明确改善" in feedback["hypothesis_evaluation"]


def test_run_auto_campaign_does_not_promote_regressed_round_to_best_parent(monkeypatch) -> None:
    service = AutoFactorMiningService()
    selection_call_count = {"count": 0}

    def fake_run_auto_mining(**kwargs):
        base_factors = list(kwargs["base_factors"])
        if base_factors == ["Alpha1"]:
            return {
                "factors": [
                    {
                        "name": "Auto_1",
                        "expression": "expr_1",
                        "score": 50.0,
                        "grade": "A",
                        "report_metrics": {"sharpe": 1.2},
                        "backtest_summary": {
                            "rank_ic_mean": 0.022,
                            "long_short_sharpe": 0.8,
                            "long_short_annual": 0.16,
                            "ic_ir": 0.5,
                            "turnover": 0.32,
                            "wq_fitness": 1.0,
                        },
                        "wq_brain": {"wq_rating": "A", "wq_returns": 0.16, "wq_fitness": 1.0},
                        "interpretation": {"weaknesses": ["第 1 轮问题"], "next_steps": ["第 1 轮动作"]},
                        "task_details": {
                            "round_evaluation": {
                                "base_factors": ["Alpha1"],
                                "primary_problem": "第 1 轮问题",
                                "recommended_goal": "ls_sharpe",
                                "suggested_actions": ["第 1 轮动作"],
                                "metric_snapshot": {"score": 50.0, "rank_ic": 0.022, "ls_sharpe": 0.8, "turnover": 0.32},
                            }
                        },
                    }
                ],
                "best_score": 50.0,
                "avg_score": 48.0,
                "generations": 1,
                "fitness_history": {"best": [50.0], "average": [48.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1"],
                    "primary_problem": "第 1 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 1 轮动作"],
                    "metric_snapshot": {"score": 50.0, "rank_ic": 0.022, "ls_sharpe": 0.8, "turnover": 0.32},
                },
            }
        if base_factors == ["Alpha1", "BetaFactor"]:
            return {
                "factors": [
                    {
                        "name": "Auto_2",
                        "expression": "expr_2",
                        "score": 52.0,
                        "grade": "A",
                        "report_metrics": {"sharpe": 0.95},
                        "backtest_summary": {
                            "rank_ic_mean": 0.015,
                            "long_short_sharpe": 0.73,
                            "long_short_annual": 0.15,
                            "ic_ir": 0.45,
                            "turnover": 0.45,
                            "wq_fitness": 1.0,
                        },
                        "wq_brain": {"wq_rating": "A", "wq_returns": 0.15, "wq_fitness": 1.0},
                        "interpretation": {"weaknesses": ["第 2 轮问题"], "next_steps": ["第 2 轮动作"]},
                        "task_details": {
                            "round_evaluation": {
                                "base_factors": ["Alpha1", "BetaFactor"],
                                "primary_problem": "第 2 轮问题",
                                "recommended_goal": "ls_sharpe",
                                "suggested_actions": ["第 2 轮动作"],
                                "metric_snapshot": {"score": 52.0, "rank_ic": 0.015, "ls_sharpe": 0.73, "turnover": 0.45},
                            }
                        },
                    }
                ],
                "best_score": 52.0,
                "avg_score": 50.0,
                "generations": 1,
                "fitness_history": {"best": [52.0], "average": [50.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1", "BetaFactor"],
                    "primary_problem": "第 2 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 2 轮动作"],
                    "metric_snapshot": {"score": 52.0, "rank_ic": 0.015, "ls_sharpe": 0.73, "turnover": 0.45},
                },
            }
        if base_factors == ["Alpha1", "GammaFactor"]:
            return {
                "factors": [
                    {
                        "name": "Auto_3",
                        "expression": "expr_3",
                        "score": 49.0,
                        "grade": "A",
                        "report_metrics": {"sharpe": 1.0},
                        "backtest_summary": {
                            "rank_ic_mean": 0.02,
                            "long_short_sharpe": 0.78,
                            "long_short_annual": 0.15,
                            "ic_ir": 0.48,
                            "turnover": 0.34,
                            "wq_fitness": 1.0,
                        },
                        "wq_brain": {"wq_rating": "A", "wq_returns": 0.15, "wq_fitness": 1.0},
                        "interpretation": {"weaknesses": ["第 3 轮问题"], "next_steps": ["第 3 轮动作"]},
                        "task_details": {
                            "round_evaluation": {
                                "base_factors": ["Alpha1", "GammaFactor"],
                                "primary_problem": "第 3 轮问题",
                                "recommended_goal": "ls_sharpe",
                                "suggested_actions": ["第 3 轮动作"],
                                "metric_snapshot": {"score": 49.0, "rank_ic": 0.02, "ls_sharpe": 0.78, "turnover": 0.34},
                            }
                        },
                    }
                ],
                "best_score": 49.0,
                "avg_score": 47.0,
                "generations": 1,
                "fitness_history": {"best": [49.0], "average": [47.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1", "GammaFactor"],
                    "primary_problem": "第 3 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 3 轮动作"],
                    "metric_snapshot": {"score": 49.0, "rank_ic": 0.02, "ls_sharpe": 0.78, "turnover": 0.34},
                },
            }
        raise AssertionError(f"unexpected base factors: {base_factors}")

    def fake_select_continue_factors(**kwargs):
        selection_call_count["count"] += 1
        if selection_call_count["count"] == 1:
            return {
                "selected_factors": ["BetaFactor"],
                "selection_rationale": "第 1 轮后补因子",
                "per_factor_reason": {"BetaFactor": "第 1 轮建议"},
                "continuation_context": {
                    "primary_problem": "第 1 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 1 轮动作"],
                    "summary_text": "沿着第 1 轮问题继续优化",
                    "should_adjust_base_factors": True,
                    "selection_confidence": 3,
                },
            }
        assert list((kwargs.get("parent_request") or {}).get("base_factors") or []) == ["Alpha1"]
        return {
            "selected_factors": ["GammaFactor"],
            "selection_rationale": "继续基于未退化的最佳轮探索",
            "per_factor_reason": {"GammaFactor": "沿用第 1 轮 parent"},
            "continuation_context": {
                "primary_problem": "第 1 轮问题",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["第 1 轮动作"],
                "summary_text": "继续沿着第 1 轮问题优化",
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)

    result = service.run_auto_campaign(
        prompt="提升风险调整后收益",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=3,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 60},
    )

    assert result["rounds"][1]["continuation_feedback"]["accepted_as_best"] is False
    assert result["rounds"][1]["continuation_feedback"]["fallback_parent_strategy"] == "best_score_so_far"
    assert result["rounds"][1]["continuation_feedback"]["metric_deltas"]["ls_sharpe"] < 0
    assert result["rounds"][2]["input_base_factors"] == ["Alpha1", "GammaFactor"]
    assert result["final_round_result"]["round_evaluation"]["base_factors"] == ["Alpha1", "GammaFactor"]
    assert result["best_parent_result"]["round_evaluation"]["base_factors"] == ["Alpha1"]
    assert result["retained_factors"][0]["expression"] == "expr_3"
    assert result["best_parent_retained_factors"][0]["expression"] == "expr_1"
    assert result["latest_round_retained_factors"][0]["expression"] == "expr_3"
    assert result["best_result_retained_factors"][0]["expression"] == "expr_2"
    assert result["final_round_task_id"] == result["latest_round_task_id"]
    assert result["latest_round_result"]["round_evaluation"]["base_factors"] == ["Alpha1", "GammaFactor"]
    assert result["best_parent_task_id"] != result["final_round_task_id"]
    assert result["fitness_history"]["best"] == [50.0, 52.0, 52.0]
    assert result["fitness_history"]["average"] == [48.0, 49.0, 48.3333]


def test_run_auto_campaign_promotes_round_when_target_metric_improves_despite_small_score_drop(monkeypatch) -> None:
    service = AutoFactorMiningService()
    selection_call_count = {"count": 0}

    monkeypatch.setattr(
        service,
        "resolve_base_factor_codes",
        lambda base_factors: [
            {
                "Alpha1": "close/open",
                "BetaFactor": "beta_factor",
                "GammaFactor": "gamma_factor",
            }.get(name, name.lower())
            for name in base_factors
        ],
    )

    def fake_run_auto_mining(**kwargs):
        base_factors = list(kwargs["base_factors"])
        if base_factors == ["Alpha1"]:
            return {
                "factors": [
                    {
                        "name": "Auto_1",
                        "expression": "expr_1",
                        "score": 60.0,
                        "grade": "A",
                        "report_metrics": {"sharpe": 0.95},
                        "backtest_summary": {
                            "rank_ic_mean": 0.022,
                            "long_short_sharpe": 0.78,
                            "long_short_annual": 0.16,
                            "ic_ir": 0.5,
                            "turnover": 0.32,
                            "wq_fitness": 1.0,
                        },
                        "wq_brain": {"wq_rating": "A", "wq_returns": 0.16, "wq_fitness": 1.0},
                        "interpretation": {"weaknesses": ["第 1 轮问题"], "next_steps": ["第 1 轮动作"]},
                        "task_details": {
                            "round_evaluation": {
                                "base_factors": ["Alpha1"],
                                "primary_problem": "第 1 轮问题",
                                "recommended_goal": "ls_sharpe",
                                "suggested_actions": ["第 1 轮动作"],
                                "metric_snapshot": {"score": 60.0, "rank_ic": 0.022, "ls_sharpe": 0.78, "turnover": 0.32},
                            }
                        },
                    }
                ],
                "best_score": 60.0,
                "avg_score": 58.0,
                "generations": 1,
                "fitness_history": {"best": [60.0], "average": [58.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1"],
                    "primary_problem": "第 1 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 1 轮动作"],
                    "metric_snapshot": {"score": 60.0, "rank_ic": 0.022, "ls_sharpe": 0.78, "turnover": 0.32},
                },
            }
        if base_factors == ["Alpha1", "BetaFactor"]:
            return {
                "factors": [
                    {
                        "name": "Auto_2",
                        "expression": "rank(ts_mean(close/open, 5) * ts_mean(beta_factor, 5))",
                        "score": 59.4,
                        "grade": "A",
                        "report_metrics": {"sharpe": 1.04},
                        "backtest_summary": {
                            "rank_ic_mean": 0.0212,
                            "long_short_sharpe": 0.86,
                            "long_short_annual": 0.169,
                            "ic_ir": 0.52,
                            "turnover": 0.34,
                            "wq_fitness": 1.02,
                        },
                        "wq_brain": {"wq_rating": "A", "wq_returns": 0.169, "wq_fitness": 1.02},
                        "interpretation": {"weaknesses": ["第 2 轮问题"], "next_steps": ["第 2 轮动作"]},
                        "task_details": {
                            "round_evaluation": {
                                "base_factors": ["Alpha1", "BetaFactor"],
                                "primary_problem": "第 2 轮问题",
                                "recommended_goal": "ls_sharpe",
                                "suggested_actions": ["第 2 轮动作"],
                                "metric_snapshot": {"score": 59.4, "rank_ic": 0.0212, "ls_sharpe": 0.86, "turnover": 0.34},
                                "parent_expression": "rank(ts_mean(close/open, 5))",
                            }
                        },
                    }
                ],
                "best_score": 59.4,
                "avg_score": 57.4,
                "generations": 1,
                "fitness_history": {"best": [59.4], "average": [57.4]},
                "round_evaluation": {
                    "base_factors": ["Alpha1", "BetaFactor"],
                    "primary_problem": "第 2 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 2 轮动作"],
                    "metric_snapshot": {"score": 59.4, "rank_ic": 0.0212, "ls_sharpe": 0.86, "turnover": 0.34},
                },
            }
        if base_factors == ["Alpha1", "BetaFactor", "GammaFactor"]:
            return {
                "factors": [
                    {
                        "name": "Auto_3",
                        "expression": "rank(ts_mean(close/open, 5) * ts_mean(beta_factor, 5) * ts_mean(gamma_factor, 5))",
                        "score": 63.0,
                        "grade": "A",
                        "report_metrics": {"sharpe": 1.08},
                        "backtest_summary": {
                            "rank_ic_mean": 0.024,
                            "long_short_sharpe": 0.9,
                            "long_short_annual": 0.175,
                            "ic_ir": 0.54,
                            "turnover": 0.33,
                            "wq_fitness": 1.05,
                        },
                        "wq_brain": {"wq_rating": "A", "wq_returns": 0.175, "wq_fitness": 1.05},
                        "interpretation": {"weaknesses": ["第 3 轮问题"], "next_steps": ["第 3 轮动作"]},
                        "task_details": {
                            "round_evaluation": {
                                "base_factors": ["Alpha1", "BetaFactor", "GammaFactor"],
                                "primary_problem": "第 3 轮问题",
                                "recommended_goal": "ls_sharpe",
                                "suggested_actions": ["第 3 轮动作"],
                                "metric_snapshot": {"score": 63.0, "rank_ic": 0.024, "ls_sharpe": 0.9, "turnover": 0.33},
                                "parent_expression": "rank(ts_mean(close/open, 5) * ts_mean(beta_factor, 5))",
                            }
                        },
                    }
                ],
                "best_score": 63.0,
                "avg_score": 61.0,
                "generations": 1,
                "fitness_history": {"best": [63.0], "average": [61.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1", "BetaFactor", "GammaFactor"],
                    "primary_problem": "第 3 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 3 轮动作"],
                    "metric_snapshot": {"score": 63.0, "rank_ic": 0.024, "ls_sharpe": 0.9, "turnover": 0.33},
                },
            }
        raise AssertionError(f"unexpected base factors: {base_factors}")

    def fake_select_continue_factors(**kwargs):
        selection_call_count["count"] += 1
        if selection_call_count["count"] == 1:
            return {
                "selected_factors": ["BetaFactor"],
                "selection_rationale": "第 1 轮后补因子",
                "per_factor_reason": {"BetaFactor": "第 1 轮建议"},
                "continuation_context": {
                    "primary_problem": "第 1 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 1 轮动作"],
                    "summary_text": "沿着第 1 轮问题继续优化",
                    "should_adjust_base_factors": True,
                    "selection_confidence": 3,
                },
            }
        assert list((kwargs.get("parent_request") or {}).get("base_factors") or []) == ["Alpha1", "BetaFactor"]
        return {
            "selected_factors": ["GammaFactor"],
            "selection_rationale": "继续基于已升级的第 2 轮探索",
            "per_factor_reason": {"GammaFactor": "沿用第 2 轮 parent"},
            "continuation_context": {
                "primary_problem": "第 2 轮问题",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["第 2 轮动作"],
                "summary_text": "继续沿着第 2 轮问题优化",
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)

    result = service.run_auto_campaign(
        prompt="提升风险调整后收益",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=3,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 0},
    )

    assert result["rounds"][1]["continuation_feedback"]["accepted_as_best"] is True
    assert result["rounds"][1]["continuation_feedback"]["fallback_parent_strategy"] is None
    assert result["rounds"][1]["continuation_feedback"]["metric_deltas"]["score"] < 0
    assert result["rounds"][1]["continuation_feedback"]["metric_deltas"]["ls_sharpe"] > 0
    assert result["rounds"][2]["input_base_factors"] == ["Alpha1", "BetaFactor", "GammaFactor"]
    assert result["rounds"][2]["continuation_hypothesis"]["reason"] == "第 2 轮问题"
    assert result["rounds"][2]["continuation_hypothesis"]["target_goal"] == "ls_sharpe"
    assert result["best_parent_result"]["round_evaluation"]["base_factors"] == ["Alpha1", "BetaFactor", "GammaFactor"]
    assert result["final_round_result"]["round_evaluation"]["base_factors"] == ["Alpha1", "BetaFactor", "GammaFactor"]


def test_run_auto_campaign_forces_small_step_exploration_after_regression_even_if_best_parent_holds(monkeypatch) -> None:
    service = AutoFactorMiningService()
    selection_call_count = {"count": 0}

    def fake_run_auto_mining(**kwargs):
        base_factors = list(kwargs["base_factors"])
        if base_factors == ["Alpha1"]:
            return {
                "factors": [
                    {
                        "name": "Auto_1",
                        "expression": "expr_1",
                        "score": 60.0,
                        "grade": "A",
                        "report_metrics": {"sharpe": 1.1},
                        "backtest_summary": {
                            "rank_ic_mean": 0.022,
                            "long_short_sharpe": 0.9,
                            "long_short_annual": 0.18,
                            "ic_ir": 0.5,
                            "turnover": 0.34,
                            "wq_fitness": 1.0,
                        },
                        "wq_brain": {"wq_rating": "A", "wq_returns": 0.18, "wq_fitness": 1.0},
                        "interpretation": {"weaknesses": ["整体指标较均衡，但仍可围绕目标继续精修。"], "next_steps": ["第 1 轮动作"]},
                        "task_details": {
                            "round_evaluation": {
                                "base_factors": ["Alpha1"],
                                "primary_problem": "整体指标较均衡，但仍可围绕目标继续精修。",
                                "recommended_goal": "ls_sharpe",
                                "suggested_actions": ["第 1 轮动作"],
                                "metric_snapshot": {"score": 60.0, "rank_ic": 0.022, "ls_sharpe": 0.9, "turnover": 0.34},
                            }
                        },
                    }
                ],
                "best_score": 60.0,
                "avg_score": 58.0,
                "generations": 1,
                "fitness_history": {"best": [60.0], "average": [58.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1"],
                    "primary_problem": "整体指标较均衡，但仍可围绕目标继续精修。",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 1 轮动作"],
                    "metric_snapshot": {"score": 60.0, "rank_ic": 0.022, "ls_sharpe": 0.9, "turnover": 0.34},
                },
            }
        if base_factors == ["Alpha1", "BetaFactor"]:
            return {
                "factors": [
                    {
                        "name": "Auto_2",
                        "expression": "expr_2",
                        "score": 40.0,
                        "grade": "B",
                        "report_metrics": {"sharpe": 0.6},
                        "backtest_summary": {
                            "rank_ic_mean": 0.01,
                            "long_short_sharpe": 0.4,
                            "long_short_annual": 0.08,
                            "ic_ir": 0.3,
                            "turnover": 0.4,
                            "wq_fitness": 0.8,
                        },
                        "wq_brain": {"wq_rating": "B", "wq_returns": 0.08, "wq_fitness": 0.8},
                        "interpretation": {"weaknesses": ["第 2 轮问题"], "next_steps": ["第 2 轮动作"]},
                        "task_details": {
                            "round_evaluation": {
                                "base_factors": ["Alpha1", "BetaFactor"],
                                "primary_problem": "第 2 轮问题",
                                "recommended_goal": "ls_sharpe",
                                "suggested_actions": ["第 2 轮动作"],
                                "metric_snapshot": {"score": 40.0, "rank_ic": 0.01, "ls_sharpe": 0.4, "turnover": 0.4},
                            }
                        },
                    }
                ],
                "best_score": 40.0,
                "avg_score": 38.0,
                "generations": 1,
                "fitness_history": {"best": [40.0], "average": [38.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1", "BetaFactor"],
                    "primary_problem": "第 2 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 2 轮动作"],
                    "metric_snapshot": {"score": 40.0, "rank_ic": 0.01, "ls_sharpe": 0.4, "turnover": 0.4},
                },
            }
        if base_factors == ["Alpha1", "GammaFactor"]:
            return {
                "factors": [
                    {
                        "name": "Auto_3",
                        "expression": "expr_3",
                        "score": 45.0,
                        "grade": "B",
                        "report_metrics": {"sharpe": 0.8},
                        "backtest_summary": {
                            "rank_ic_mean": 0.015,
                            "long_short_sharpe": 0.55,
                            "long_short_annual": 0.11,
                            "ic_ir": 0.35,
                            "turnover": 0.36,
                            "wq_fitness": 0.85,
                        },
                        "wq_brain": {"wq_rating": "B", "wq_returns": 0.11, "wq_fitness": 0.85},
                        "interpretation": {"weaknesses": ["第 3 轮问题"], "next_steps": ["第 3 轮动作"]},
                        "task_details": {
                            "round_evaluation": {
                                "base_factors": ["Alpha1", "GammaFactor"],
                                "primary_problem": "第 3 轮问题",
                                "recommended_goal": "ls_sharpe",
                                "suggested_actions": ["第 3 轮动作"],
                                "metric_snapshot": {"score": 45.0, "rank_ic": 0.015, "ls_sharpe": 0.55, "turnover": 0.36},
                            }
                        },
                    }
                ],
                "best_score": 45.0,
                "avg_score": 43.0,
                "generations": 1,
                "fitness_history": {"best": [45.0], "average": [43.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1", "GammaFactor"],
                    "primary_problem": "第 3 轮问题",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 3 轮动作"],
                    "metric_snapshot": {"score": 45.0, "rank_ic": 0.015, "ls_sharpe": 0.55, "turnover": 0.36},
                },
            }
        raise AssertionError(f"unexpected base factors: {base_factors}")

    def fake_select_continue_factors(**kwargs):
        selection_call_count["count"] += 1
        if selection_call_count["count"] == 1:
            return {
                "selected_factors": ["BetaFactor"],
                "selection_rationale": "第一轮后探索一个新因子",
                "per_factor_reason": {"BetaFactor": "第 1 轮建议"},
                "continuation_context": {
                    "primary_problem": "整体指标较均衡，但仍可围绕目标继续精修。",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["第 1 轮动作"],
                    "summary_text": "保持当前基础因子组合，优先微调表达式",
                    "should_adjust_base_factors": True,
                    "selection_confidence": 3,
                },
            }
        return {
            "selected_factors": ["GammaFactor"],
            "selection_rationale": "最佳轮本来想 hold，但提供一个备选探索因子",
            "per_factor_reason": {"GammaFactor": "受控小步探索"},
            "continuation_context": {
                "primary_problem": "整体指标较均衡，但仍可围绕目标继续精修。",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["第 1 轮动作"],
                "summary_text": "保持当前基础因子组合，优先微调表达式",
                "should_adjust_base_factors": False,
                "hold_reason": "上一轮没有暴露出足够明确的结构性短板，优先保持当前基础因子组合，先在表达式结构上做微调。",
                "selection_confidence": 0,
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_continue_factors", fake_select_continue_factors)

    result = service.run_auto_campaign(
        prompt="提升风险调整后收益",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=3,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 0},
    )

    assert result["rounds"][1]["continuation_feedback"]["fallback_parent_strategy"] == "best_score_so_far"
    assert result["rounds"][1]["continuation_hypothesis"]["next_round_forced_exploration"] is True
    assert "受控的小步探索" in result["rounds"][1]["continuation_hypothesis"]["next_round_forced_exploration_reason"]
    assert result["rounds"][2]["input_base_factors"] == ["Alpha1", "GammaFactor"]


def test_run_auto_campaign_keeps_best_result_when_later_round_has_no_executable_candidates(monkeypatch) -> None:
    service = AutoFactorMiningService()
    call_count = {"count": 0}

    def fake_run_auto_mining(**kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return {
                "factors": [
                    {
                        "name": "Auto_1",
                        "expression": "expr_1",
                        "score": 80.0,
                        "grade": "A",
                        "report_metrics": {"sharpe": 1.1},
                        "backtest_summary": {
                            "long_short_sharpe": 1.0,
                            "long_short_annual": 0.16,
                            "rank_ic_mean": 0.02,
                            "turnover": 0.34,
                        },
                    }
                ],
                "best_score": 80.0,
                "avg_score": 78.0,
                "generations": 1,
                "fitness_history": {"best": [80.0], "average": [78.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1"],
                    "primary_problem": "第 1 轮问题",
                    "recommended_goal": "score",
                    "suggested_actions": ["第 1 轮动作"],
                    "metric_snapshot": {"score": 80.0},
                },
            }
        raise ValueError("未生成可执行候选表达式")

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(
        service,
        "select_continue_factors",
        lambda **kwargs: {
            "selected_factors": ["BetaFactor"],
            "selection_rationale": "补一个新因子",
            "per_factor_reason": {"BetaFactor": "用于下一轮探索"},
            "continuation_context": {
                "primary_problem": "第 1 轮问题",
                "recommended_goal": "score",
                "suggested_actions": ["第 1 轮动作"],
                "summary_text": "沿着第 1 轮问题继续优化",
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
            },
        },
    )

    result = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 0},
    )

    assert result["completed_with_failures"] is True
    assert result["failure_reason"] == "未生成可执行候选表达式"
    assert result["final_round_result"]["best_score"] == 80.0
    assert result["best_parent_result"]["best_score"] == 80.0
    assert result["latest_round_result"] is None
    assert result["latest_round_task_id"].startswith("campaign-")
    assert result["rounds"][-1]["status"] == "failed"
    assert result["rounds"][-1]["error"] == "未生成可执行候选表达式"
    assert result["rounds"][-1]["continuation_feedback"]["accepted_as_best"] is False


def test_run_auto_campaign_returns_structured_failure_when_first_round_has_no_executable_candidates(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        service,
        "run_auto_mining",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("未生成可执行候选表达式")),
    )

    result = service.run_auto_campaign(
        prompt="提升综合分数",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="score",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 0},
    )

    assert result["completed_with_failures"] is True
    assert result["failure_reason"] == "未生成可执行候选表达式"
    assert result["best_score"] == 0.0
    assert result["final_round_result"] is None
    assert result["best_parent_result"] is None
    assert result["latest_round_result"] is None
    assert len(result["rounds"]) == 1
    assert result["rounds"][0]["status"] == "failed"
    assert result["rounds"][0]["input_base_factors"] == ["Alpha1"]


def test_run_auto_campaign_keeps_round_evaluation_and_display_scores_consistent(monkeypatch) -> None:
    service = AutoFactorMiningService()
    call_count = {"count": 0}

    def fake_run_auto_mining(**kwargs):
        call_count["count"] += 1
        round_index = call_count["count"]
        if round_index == 1:
            best_score = 80.0
            avg_score = 76.0
            ls_sharpe = 0.42
            base_factors = ["Alpha1"]
        else:
            best_score = 70.0
            avg_score = 68.0
            ls_sharpe = 0.39
            base_factors = ["Alpha1", "BetaFactor"]

        factor = {
            "name": f"Factor_{round_index}",
            "expression": f"expr_{round_index}",
            "score": best_score,
            "report_metrics": {"sharpe": ls_sharpe},
            "backtest_summary": {
                "long_short_sharpe": ls_sharpe,
                "long_short_annual": 0.12,
                "rank_ic_mean": 0.02,
                "turnover": 0.35,
            },
            "task_details": {
                "round_evaluation": {
                    "base_factors": list(base_factors),
                    "primary_problem": f"第 {round_index} 轮问题",
                    "secondary_problem": "",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": [f"第 {round_index} 轮建议"],
                    "metric_snapshot": {"score": best_score, "ls_sharpe": ls_sharpe},
                }
            },
        }
        return {
            "factors": [factor],
            "best_score": best_score,
            "avg_score": avg_score,
            "fitness_history": {"best": [best_score], "average": [avg_score]},
            "round_evaluation": {
                "base_factors": list(base_factors),
                "primary_problem": f"第 {round_index} 轮问题",
                "secondary_problem": "",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": [f"第 {round_index} 轮建议"],
                "metric_snapshot": {"score": best_score, "ls_sharpe": ls_sharpe},
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(
        service,
        "select_continue_factors",
        lambda **kwargs: {
            "selected_factors": ["BetaFactor"],
            "selection_rationale": "补充 Beta 因子",
            "per_factor_reason": {"BetaFactor": "补充风格暴露"},
            "continuation_context": {
                "primary_problem": "第 1 轮问题",
                "secondary_problem": "",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["第 1 轮建议"],
                "summary_text": "第 1 轮问题",
                "selection_instructions": "继续提升 ls_sharpe",
                "should_adjust_base_factors": True,
                "hold_reason": "",
                "selection_confidence": 2,
                "replace_base_factors": [],
            },
        },
    )

    result = service.run_auto_campaign(
        prompt="提升风险调整后收益",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 0},
    )

    assert result["best_score"] == 80.0
    assert result["avg_score"] == 72.0
    assert result["campaign_metric"] == "ls_sharpe"
    assert result["fitness_history"] == {"best": [80.0, 80.0], "average": [76.0, 72.0]}
    assert result["rounds"][0]["round_evaluation"]["primary_problem"] == "第 1 轮问题"
    assert result["rounds"][1]["round_evaluation"]["primary_problem"] == "第 2 轮问题"
    assert result["final_round_result"]["round_evaluation"]["primary_problem"] == "第 2 轮问题"


def test_run_auto_campaign_selection_rationale_uses_previous_round_analysis(monkeypatch) -> None:
    service = AutoFactorMiningService()
    call_count = {"count": 0}

    def fake_run_auto_mining(**kwargs):
        call_count["count"] += 1
        round_index = call_count["count"]
        if round_index == 1:
            base_factors = ["close_open_ratio", "high_low_ratio"]
            primary_problem = "换手率偏高（0.73），真实可交易性承压，需要优先压低噪声和频繁切换。"
            secondary_problem = "L/S Sharpe 偏低（2.32），收益质量仍需改善。"
            score = 74.17
            ls_sharpe = 2.3165
            rank_ic = 0.032638
            turnover = 0.731077
        else:
            base_factors = ["close_open_ratio", "high_low_ratio", "price_vwma_ratio"]
            primary_problem = "L/S Sharpe 偏低（2.69），收益质量仍需改善。"
            secondary_problem = "整体指标较均衡，但仍可围绕目标继续精修。"
            score = 67.51
            ls_sharpe = 2.6879
            rank_ic = 0.021083
            turnover = 0.3837

        factor = {
            "name": f"Factor_{round_index}",
            "expression": f"expr_{round_index}",
            "score": score,
            "report_metrics": {"sharpe": ls_sharpe},
            "backtest_summary": {
                "long_short_sharpe": ls_sharpe,
                "long_short_annual": 0.12,
                "rank_ic_mean": rank_ic,
                "turnover": turnover,
            },
            "task_details": {
                "round_evaluation": {
                    "base_factors": list(base_factors),
                    "primary_problem": primary_problem,
                    "secondary_problem": secondary_problem,
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": [f"第 {round_index} 轮建议"],
                    "metric_snapshot": {"score": score, "ls_sharpe": ls_sharpe, "rank_ic": rank_ic, "turnover": turnover},
                }
            },
        }
        return {
            "factors": [factor],
            "best_score": score,
            "avg_score": score,
            "fitness_history": {"best": [score], "average": [score]},
            "round_evaluation": factor["task_details"]["round_evaluation"],
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(
        service,
        "select_continue_factors",
        lambda **kwargs: {
            "selected_factors": ["price_vwma_ratio"],
            "selection_rationale": "补一个平滑量价因子",
            "per_factor_reason": {"price_vwma_ratio": "补充平滑量价结构"},
            "continuation_context": {
                "primary_problem": "换手率偏高（0.73），真实可交易性承压，需要优先压低噪声和频繁切换。",
                "secondary_problem": "L/S Sharpe 偏低（2.32），收益质量仍需改善。",
                "recommended_goal": "ls_sharpe",
                "selection_instructions": "优先补充更平滑的量价因子。",
                "preferred_keywords": ["ema", "sma"],
                "avoid_keywords": ["breakout"],
                "selection_confidence": 4,
                "should_adjust_base_factors": True,
                "hold_reason": "",
                "replace_base_factors": [],
            },
        },
    )

    result = service.run_auto_campaign(
        prompt="提升量价复合因子的稳定性与横截面区分度",
        base_factors=["close_open_ratio", "high_low_ratio"],
        start_date="2024-01-01",
        end_date="2024-03-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "any", "score_min": 0},
    )

    second_round = result["rounds"][1]
    assert "换手率偏高" in result["rounds"][0]["selection_rationale"]
    assert "换手率偏高" in second_round["selection_rationale"]
    assert "L/S Sharpe 偏低（2.69）" not in second_round["selection_rationale"]
    assert second_round["per_factor_reason"]["price_vwma_ratio"] == "围绕“ls_sharpe”保留或补充该基础因子。"


def test_run_auto_campaign_persists_continuation_instructions_in_round_hypothesis(monkeypatch) -> None:
    service = AutoFactorMiningService()

    def fake_run_auto_mining(**kwargs):
        round_index = 1 if kwargs["base_factors"] == ["Alpha1"] else 2
        return {
            "factors": [
                {
                    "name": f"Auto_{round_index}",
                    "expression": f"expr_{round_index}",
                    "score": 50.0 - round_index,
                    "grade": "B",
                    "report_metrics": {"sharpe": 0.5},
                    "backtest_summary": {"long_short_sharpe": 0.5, "long_short_annual": 0.08, "rank_ic_mean": 0.01, "turnover": 0.4, "wq_fitness": 0.5},
                    "interpretation": {"weaknesses": [f"第 {round_index} 轮问题"], "next_steps": [f"第 {round_index} 轮动作"]},
                    "task_details": {
                        "round_evaluation": {
                            "base_factors": list(kwargs["base_factors"]),
                            "primary_problem": f"第 {round_index} 轮问题",
                            "secondary_problem": "次要问题",
                            "recommended_goal": "ls_sharpe",
                            "selection_instructions": f"第 {round_index} 轮指令",
                            "preferred_keywords": ["quality"],
                            "avoid_keywords": ["breakout"],
                            "selection_confidence": 3,
                            "should_adjust_base_factors": True,
                            "hold_reason": "",
                            "metric_snapshot": {"score": 50.0 - round_index},
                        }
                    },
                }
            ],
            "best_score": 50.0 - round_index,
            "avg_score": 50.0 - round_index,
            "fitness_history": {"best": [50.0 - round_index], "average": [50.0 - round_index]},
            "round_evaluation": {
                "base_factors": list(kwargs["base_factors"]),
                "primary_problem": f"第 {round_index} 轮问题",
                "secondary_problem": "次要问题",
                "recommended_goal": "ls_sharpe",
                "selection_instructions": f"第 {round_index} 轮指令",
                "preferred_keywords": ["quality"],
                "avoid_keywords": ["breakout"],
                "selection_confidence": 3,
                "should_adjust_base_factors": True,
                "hold_reason": "",
                "metric_snapshot": {"score": 50.0 - round_index},
            },
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(
        service,
        "select_continue_factors",
        lambda **kwargs: {
            "selected_factors": ["Beta1"],
            "selection_rationale": "ok",
            "per_factor_reason": {"Beta1": "ok"},
            "continuation_context": {
                "primary_problem": "第 1 轮问题",
                "secondary_problem": "次要问题",
                "recommended_goal": "ls_sharpe",
                "selection_instructions": "第 1 轮指令",
                "preferred_keywords": ["quality"],
                "avoid_keywords": ["breakout"],
                "selection_confidence": 3,
                "should_adjust_base_factors": True,
                "hold_reason": "",
            },
        },
    )

    result = service.run_auto_campaign(
        prompt="提升风险调整后收益",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=2,
        n_candidates_per_round=1,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 0},
    )

    hypothesis = result["rounds"][1]["continuation_hypothesis"]
    assert hypothesis["reason"] == "第 1 轮问题"
    assert "本轮只围绕主要短板“第 1 轮问题”做一次受控迭代" in hypothesis["selection_instructions"]
    assert "quality" in hypothesis["preferred_keywords"]
    assert "breakout" in hypothesis["avoid_keywords"]


def test_run_auto_campaign_uses_continuation_seed_expression_when_best_parent_not_upgraded(monkeypatch) -> None:
    service = AutoFactorMiningService()
    seen_parent_expressions: list[str] = []
    call_count = {"count": 0}

    def fake_run_auto_mining(**kwargs):
        call_count["count"] += 1
        continuation_context = kwargs.get("continuation_context") or {}
        seen_parent_expressions.append(str(continuation_context.get("parent_expression") or ""))
        if call_count["count"] == 1:
            return {
                "factors": [
                    {
                        "name": "Round1Best",
                        "expression": "rank(ts_mean(close/open,5))",
                        "score": 20.0,
                        "grade": "B",
                        "report_metrics": {"sharpe": 0.8},
                        "backtest_summary": {
                            "long_short_sharpe": 0.9,
                            "long_short_annual": 0.12,
                            "rank_ic_mean": 0.01,
                            "turnover": 0.32,
                        },
                    }
                ],
                "best_score": 20.0,
                "avg_score": 20.0,
                "generations": 1,
                "fitness_history": {"best": [20.0], "average": [20.0]},
                "round_evaluation": {
                    "base_factors": ["Alpha1"],
                    "primary_problem": "横截面 rankIC 偏弱",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["优先引入新增量价因子"],
                    "metric_snapshot": {"score": 20.0, "rank_ic": 0.01, "ls_sharpe": 0.9, "turnover": 0.32},
                },
            }
        if call_count["count"] == 2:
            return {
                "factors": [
                    {
                        "name": "Round2Smooth",
                        "expression": "rank(ts_mean(rank(ts_mean(close/open,5)),3))",
                        "score": 22.0,
                        "grade": "B",
                        "report_metrics": {"sharpe": 0.7},
                        "backtest_summary": {
                            "long_short_sharpe": 0.7,
                            "long_short_annual": 0.10,
                            "rank_ic_mean": 0.011,
                            "turnover": 0.35,
                        },
                    },
                    {
                        "name": "Round2Seed",
                        "expression": "rank(ts_mean(rank(ts_mean(close/open,5)), 5) * ts_mean(force_index_ma, 5))",
                        "score": 18.5,
                        "grade": "C",
                        "report_metrics": {"sharpe": 0.65},
                        "backtest_summary": {
                            "long_short_sharpe": 0.65,
                            "long_short_annual": 0.09,
                            "rank_ic_mean": 0.013,
                            "turnover": 0.33,
                        },
                    },
                ],
                "best_score": 22.0,
                "avg_score": 20.25,
                "generations": 2,
                "fitness_history": {"best": [22.0], "average": [20.25]},
                "round_evaluation": {
                    "base_factors": ["Alpha1", "ForceIndex"],
                    "primary_problem": "横截面 rankIC 偏弱",
                    "recommended_goal": "ls_sharpe",
                    "suggested_actions": ["继续沿新增量价因子做小步探索"],
                    "metric_snapshot": {"score": 22.0, "rank_ic": 0.011, "ls_sharpe": 0.7, "turnover": 0.35},
                },
            }
        return {
            "factors": [
                {
                    "name": "Round3Best",
                    "expression": "rank(ts_mean(rank(ts_mean(close/open,5)), 5) * ts_mean(force_index_ma, 10))",
                    "score": 21.5,
                    "grade": "B",
                    "report_metrics": {"sharpe": 0.72},
                    "backtest_summary": {
                        "long_short_sharpe": 0.72,
                        "long_short_annual": 0.11,
                        "rank_ic_mean": 0.014,
                        "turnover": 0.31,
                    },
                }
            ],
            "best_score": 21.5,
            "avg_score": 21.5,
            "generations": 1,
            "fitness_history": {"best": [21.5], "average": [21.5]},
            "round_evaluation": {
                "base_factors": ["Alpha1", "ForceIndex"],
                "primary_problem": "横截面 rankIC 偏弱",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["继续沿新增量价因子做小步探索"],
                "metric_snapshot": {"score": 21.5, "rank_ic": 0.014, "ls_sharpe": 0.72, "turnover": 0.31},
            },
        }

    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda factors: ["close/open"] if factors == ["Alpha1"] else ["close/open", "force_index_ma"])
    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(
        service,
        "select_continue_factors",
        lambda **kwargs: {
            "selected_factors": ["ForceIndex"],
            "selection_rationale": "补入新因子",
            "per_factor_reason": {"ForceIndex": "增强量价信息"},
            "continuation_context": {
                "primary_problem": "横截面 rankIC 偏弱",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["继续沿新增量价因子做小步探索"],
                "summary_text": "继续探索新增因子",
                "should_adjust_base_factors": True,
                "selection_confidence": 3,
            },
        },
    )

    result = service.run_auto_campaign(
        prompt="提升风险调整后收益",
        base_factors=["Alpha1"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        exploration_rounds=3,
        n_candidates_per_round=2,
        additional_factor_count_per_round=1,
        factor_update_mode="append",
        parent_selection_strategy="best_score_so_far",
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        retention_filter={"match_mode": "all", "score_min": 0},
    )

    assert result["rounds"][1]["continuation_feedback"]["accepted_as_best"] is False
    assert result["rounds"][1]["continuation_seed_factor"]["expression"] == "rank(ts_mean(rank(ts_mean(close/open,5)), 5) * ts_mean(force_index_ma, 5))"
    assert seen_parent_expressions[2] == "rank(ts_mean(rank(ts_mean(close/open,5)), 5) * ts_mean(force_index_ma, 5))"


def test_select_continue_factors_passes_parent_context_to_llm_selection(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        service,
        "select_factors",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) or {
            "selected_factors": ["Beta"],
            "selection_rationale": "ok",
            "per_factor_reason": {"Beta": "ok"},
        },
    )

    result = service.select_continue_factors(
        parent_result={
            "round_evaluation": {
                "base_factors": ["Alpha"],
                "primary_problem": "Sharpe 偏低",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["补充波动率因子"],
                "metric_snapshot": {"score": 61},
            },
            "factors": [
                {
                    "score": 61,
                    "report_metrics": {"sharpe": 0.5},
                    "backtest_summary": {"long_short_sharpe": 0.5, "long_short_annual": 0.08},
                    "interpretation": {"weaknesses": ["Sharpe 偏低"], "next_steps": ["补充波动率因子"]},
                }
            ],
        },
        parent_request={
            "base_factors": ["Alpha"],
            "direction": "ls_sharpe",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "universe": "hs300",
            "benchmark": "hs300",
        },
        prompt="提升风险调整后收益",
        direction="ls_sharpe",
        factor_update_mode="append",
        max_factor_count=3,
        candidate_limit=50,
    )

    assert captured["kwargs"]["direction"] == "ls_sharpe"
    assert captured["kwargs"]["start_date"] == "2024-01-01"
    assert captured["kwargs"]["end_date"] == "2024-12-31"
    assert captured["kwargs"]["universe"] == "hs300"
    assert captured["kwargs"]["benchmark"] == "hs300"
    assert "Sharpe 偏低" in captured["kwargs"]["extra_context"]
    assert "continuation_context" in captured["kwargs"]
    assert "selection_instructions" in captured["kwargs"]["continuation_context"]
    assert result["continuation_context"]["recommended_goal"] == "ls_sharpe"


def test_select_continue_factors_excludes_current_campaign_base_factors_when_parent_rolls_back(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        service,
        "select_factors",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) or {
            "selected_factors": ["Gamma"],
            "selection_rationale": "ok",
            "per_factor_reason": {"Gamma": "ok"},
        },
    )

    service.select_continue_factors(
        parent_result={
            "round_evaluation": {
                "base_factors": ["Alpha"],
                "primary_problem": "Sharpe 偏低",
                "recommended_goal": "ls_sharpe",
                "suggested_actions": ["补充波动率因子"],
                "metric_snapshot": {"score": 61},
            },
            "factors": [
                {
                    "score": 61,
                    "report_metrics": {"sharpe": 0.5},
                    "backtest_summary": {"long_short_sharpe": 0.5, "long_short_annual": 0.08},
                    "interpretation": {"weaknesses": ["Sharpe 偏低"], "next_steps": ["补充波动率因子"]},
                }
            ],
        },
        parent_request={
            "base_factors": ["Alpha"],
            "direction": "ls_sharpe",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "universe": "hs300",
            "benchmark": "hs300",
        },
        prompt="提升风险调整后收益",
        direction="ls_sharpe",
        factor_update_mode="append",
        max_factor_count=2,
        candidate_limit=20,
        current_base_factors=["Alpha", "Beta"],
    )

    assert captured["kwargs"]["exclude_factors"] == ["Alpha", "Beta"]


def test_run_auto_mining_keeps_parent_seed_continuation_when_base_factors_hold(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda factors: ["close/open", "force_index_ma"])
    monkeypatch.setattr(service.data_service, "get_stock_universe", lambda universe, date: ["000001.SZ"])
    monkeypatch.setattr(service, "_collect_sample_frames", lambda **kwargs: [])
    monkeypatch.setattr(service, "_filter_supported_expressions", lambda expressions, sample_frames, limit: expressions[:limit])

    captured_previous_expressions: list[list[str]] = []

    def fake_generate_candidate_expressions(**kwargs):
        captured_previous_expressions.append(list(kwargs.get("previous_expressions") or []))
        return [
            "rank(ts_mean(parent_seed_expr, 5))",
            "rank(ts_mean(close/open, 5))",
        ]

    monkeypatch.setattr(service, "generate_candidate_expressions", fake_generate_candidate_expressions)
    monkeypatch.setattr(
        service,
        "_build_fallback_candidate_expressions",
        lambda **kwargs: [
            "rank(ts_mean(parent_seed_expr, 5))",
            "rank(ts_mean(close/open, 5))",
        ],
    )

    def fake_evaluate_expression(**kwargs):
        expression = kwargs["expression"]
        if expression == "rank(ts_mean(parent_seed_expr, 5))":
            return SimpleNamespace(
                expression=expression,
                score=70.0,
                backtest_summary={"long_short_sharpe": 1.8, "long_short_annual": 0.22},
                report_metrics={"sharpe": 1.5},
                wq_brain={"wq_fitness": 1.7, "wq_returns": 0.2},
                interpretation={},
                execution_meta={},
            )
        return SimpleNamespace(
            expression=expression,
            score=68.0,
            backtest_summary={"long_short_sharpe": 1.4, "long_short_annual": 0.18},
            report_metrics={"sharpe": 1.2},
            wq_brain={"wq_fitness": 1.3, "wq_returns": 0.16},
            interpretation={},
            execution_meta={},
        )

    monkeypatch.setattr(service, "evaluate_expression", fake_evaluate_expression)
    monkeypatch.setattr(
        service,
        "_build_round_evaluation",
        lambda **kwargs: {
            "prompt": kwargs["prompt"],
            "base_factors": kwargs["base_factors"],
            "primary_problem": "换手率偏高",
            "recommended_goal": "ls_sharpe",
            "metric_snapshot": {"score": 70.0, "ls_sharpe": 1.8, "turnover": 0.42},
        },
    )
    monkeypatch.setattr(
        service,
        "_format_candidate_payload",
        lambda evaluation, prompt, index, base_factors, round_evaluation=None: {
            "name": f"Auto_Factor_{index + 1}",
            "expression": evaluation.expression,
            "score": evaluation.score,
            "backtest_summary": evaluation.backtest_summary,
            "report_metrics": evaluation.report_metrics,
            "wq_brain": evaluation.wq_brain,
            "interpretation": evaluation.interpretation,
            "task_details": {"round_evaluation": round_evaluation} if round_evaluation else {},
        },
    )

    result = service.run_auto_mining(
        prompt="继续围绕上一轮有效结构优化 Sharpe",
        base_factors=["Alpha1", "ForceIndex"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        n_candidates=2,
        direction="ls_sharpe",
        neutralize_industry=True,
        neutralize_cap=True,
        previous_expressions=[],
        continuation_context={
            "should_adjust_base_factors": False,
            "parent_expression": "parent_seed_expr",
            "parent_raw_expression": "parent_seed_expr",
            "recommended_goal": "ls_sharpe",
            "primary_problem": "换手率偏高",
            "previous_base_factors": ["Alpha1", "ForceIndex"],
        },
    )

    assert captured_previous_expressions
    assert "parent_seed_expr" in result["factors"][0]["expression"]


def test_select_factors_uses_real_llm_flow(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.factor_selection_service.load_factor_candidates_for_llm",
        lambda limit, selection_mode: [
            {"name": "AlphaClose", "code": "close", "category": "price", "description": "close factor"},
            {"name": "AlphaVolume", "code": "volume", "category": "volume", "description": "volume factor"},
        ],
    )
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
    )

    def fake_build_prompt(request, candidates):
        captured["request"] = request
        captured["candidates"] = candidates
        return "LLM PROMPT"

    def fake_select_with_llm(**kwargs):
        captured["llm_kwargs"] = kwargs
        return {
            "selected_factors": ["AlphaVolume", "AlphaClose"],
            "selection_rationale": "LLM 按研究目标选择了量价互补因子。",
            "per_factor_reason": {
                "AlphaVolume": "成交量维度补充趋势信息。",
                "AlphaClose": "价格维度提供主要趋势信息。",
            },
            "llm_response_id": "resp-123",
            "llm_provider": "openai_compatible",
        }

    monkeypatch.setattr("backend.services.auto_factor_mining_service.build_llm_factor_selector_prompt", fake_build_prompt)
    monkeypatch.setattr(service, "_select_factors_with_llm", fake_select_with_llm)

    result = service.select_factors(
        prompt="寻找趋势突破因子",
        direction="ls_sharpe",
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="single_stock",
        benchmark="hs300",
        max_factor_count=2,
        candidate_limit=80,
        selection_mode="manual_genetic",
        extra_context="补充控制波动率",
        exclude_factors=["AlphaClose"],
    )

    assert result["selected_factors"] == ["AlphaVolume", "AlphaClose"]
    assert "LLM PROMPT" in captured["llm_kwargs"]["prompt"]
    assert "手动遗传挖掘硬约束" in captured["llm_kwargs"]["prompt"]
    assert captured["llm_kwargs"]["prompt"].endswith("\n\n补充上下文：\n补充控制波动率")
    assert captured["llm_kwargs"]["max_factor_count"] == 2
    assert captured["llm_kwargs"]["candidates"] == [
        {"name": "AlphaVolume", "code": "volume", "category": "volume", "description": "volume factor"},
    ]
    request = captured["request"]
    assert request.prompt == "寻找趋势突破因子"
    assert request.direction == "ls_sharpe"
    assert request.start_date == "2024-01-01"
    assert request.end_date == "2024-12-31"
    assert request.universe == "single_stock"
    assert request.benchmark == "hs300"
    assert request.selection_mode == "manual_genetic"
    assert result["llm_call_mode"] == "live_api"
    assert result["llm_provider"] == "openai_compatible"
    assert result["llm_response_id"] == "resp-123"
    assert result["llm_evidence"] == {
        "call_mode": "live_api",
        "provider": "openai_compatible",
        "model": "deepseek-chat",
        "base_url": "https://example.com/v1",
        "response_id": "resp-123",
    }


def test_select_factors_prioritizes_continuation_candidates(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.factor_selection_service.load_factor_candidates_for_llm",
        lambda limit, selection_mode: [
            {
                "name": "BreakoutAlpha",
                "code": "distance_to_high_20",
                "category": "price",
                "description": "short breakout style factor",
                "snapshot_summary": {"report_metrics": {"sharpe": 0.6}, "backtest_summary": {"rank_ic_mean": 0.01}},
            },
            {
                "name": "SmoothVolumeAlpha",
                "code": "ema(volume, 10)",
                "category": "volume",
                "description": "smoothed volume trend",
                "snapshot_summary": {"report_metrics": {"sharpe": 1.2}, "backtest_summary": {"rank_ic_mean": 0.03}},
            },
            {
                "name": "VolatilityAlpha",
                "code": "std(close, 10)",
                "category": "volatility",
                "description": "medium horizon volatility control",
                "snapshot_summary": {"report_metrics": {"sharpe": 1.1}, "backtest_summary": {"rank_ic_mean": 0.025}},
            },
        ],
    )
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
    )
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.build_llm_factor_selector_prompt",
        lambda request, candidates: captured.setdefault("candidates", candidates) or "LLM PROMPT",
    )

    def fake_select_with_llm(**kwargs):
        captured["llm_prompt"] = kwargs["prompt"]
        return {
            "selected_factors": ["SmoothVolumeAlpha", "VolatilityAlpha"],
            "selection_rationale": "选择更平滑的量价与波动类因子。",
            "per_factor_reason": {
                "SmoothVolumeAlpha": "平滑量能信息有助于控制换手。",
                "VolatilityAlpha": "波动类因子可帮助约束噪声。",
            },
        }

    monkeypatch.setattr(service, "_select_factors_with_llm", fake_select_with_llm)

    continuation_context = {
        "primary_problem": "换手率偏高，可能影响真实可交易性。",
        "recommended_goal": "ls_sharpe",
        "suggested_actions": ["通过更平滑的基础因子或中周期信号压低换手。"],
        "metric_snapshot": {"turnover": 0.61, "ls_sharpe": 0.42},
        "selection_instructions": "如果上一轮问题是换手率偏高，优先选择更平滑、中周期、抗噪声的量价或波动类基础因子，避免短周期突破型因子。",
        "preferred_keywords": ["ema", "volume", "std", "volatility"],
        "avoid_keywords": ["distance_to_high", "breakout"],
    }

    result = service.select_factors(
        prompt="提升风险调整后收益",
        direction="ls_sharpe",
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        max_factor_count=2,
        candidate_limit=80,
        selection_mode="auto",
        continuation_context=continuation_context,
    )

    prioritized_candidates = captured["candidates"]
    assert [item["name"] for item in prioritized_candidates[:2]] == ["SmoothVolumeAlpha", "VolatilityAlpha"]
    assert prioritized_candidates[-1]["name"] == "BreakoutAlpha"
    assert "连续探索约束" in captured["llm_prompt"]
    assert "换手率偏高" in captured["llm_prompt"]
    assert result["selected_factors"] == ["SmoothVolumeAlpha", "VolatilityAlpha"]


def test_select_factors_penalizes_semantic_overlap_for_rankic_problem(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.factor_selection_service.load_factor_candidates_for_llm",
        lambda limit, selection_mode: [
            {
                "name": "PriceVwmaRatio",
                "code": "price_vwma_ratio",
                "category": "price",
                "description": "price ratio around vwma",
                "snapshot_summary": {"report_metrics": {"sharpe": 1.1}, "backtest_summary": {"rank_ic_mean": 0.021}},
            },
            {
                "name": "VolumeDispersionAlpha",
                "code": "ts_std(volume, 20)",
                "category": "volume",
                "description": "volume dispersion signal",
                "snapshot_summary": {"report_metrics": {"sharpe": 1.0}, "backtest_summary": {"rank_ic_mean": 0.028}},
            },
            {
                "name": "PriceVolumeCorrAlpha",
                "code": "ts_corr(close, volume, 10)",
                "category": "interaction",
                "description": "price volume interaction signal",
                "snapshot_summary": {"report_metrics": {"sharpe": 0.9}, "backtest_summary": {"rank_ic_mean": 0.024}},
            },
        ],
    )
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
    )
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.build_llm_factor_selector_prompt",
        lambda request, candidates: captured.setdefault("candidates", candidates) or "LLM PROMPT",
    )
    monkeypatch.setattr(
        service,
        "_select_factors_with_llm",
        lambda **kwargs: {
            "selected_factors": ["VolumeDispersionAlpha", "PriceVolumeCorrAlpha"],
            "selection_rationale": "优先补充新增量价交互和离散度信息。",
            "per_factor_reason": {
                "VolumeDispersionAlpha": "增加横截面离散度信息。",
                "PriceVolumeCorrAlpha": "引入新的量价交互信息。",
            },
        },
    )

    continuation_context = {
        "primary_problem": "横截面 rankIC 偏弱，说明选股区分度不足。",
        "recommended_goal": "ls_sharpe",
        "suggested_actions": ["优先补充新的量价交互或横截面离散度信息，不要重复价格比例类因子。"],
        "metric_snapshot": {"rank_ic": 0.011, "ls_sharpe": 1.02, "turnover": 0.41},
        "selection_instructions": "如果上一轮问题是横截面区分度不足，优先补充更能提升排序能力、量价交互或横截面离散度的信息源。",
        "preferred_keywords": ["volume", "interaction", "corr", "dispersion", "std"],
        "avoid_keywords": ["vwma", "ratio"],
        "base_factor_components": {
            "fields": ["close", "open", "high", "low"],
            "tokens": ["close_open_ratio", "high_low_ratio", "close", "open", "high", "low", "ratio"],
        },
        "recent_expression_components": {
            "fields": ["close", "open", "high", "low"],
            "operators": ["rank", "ts_rank"],
        },
    }

    result = service.select_factors(
        prompt="提升横截面排序能力",
        direction="ls_sharpe",
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        max_factor_count=2,
        candidate_limit=80,
        selection_mode="auto",
        continuation_context=continuation_context,
    )

    prioritized_candidates = captured["candidates"]
    assert [item["name"] for item in prioritized_candidates[:2]] == [
        "VolumeDispersionAlpha",
        "PriceVolumeCorrAlpha",
    ]
    assert prioritized_candidates[-1]["name"] == "PriceVwmaRatio"
    assert result["selected_factors"] == ["VolumeDispersionAlpha", "PriceVolumeCorrAlpha"]


def test_select_factors_requires_llm_api_key(monkeypatch) -> None:
    service = AutoFactorMiningService()
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.factor_selection_service.load_factor_candidates_for_llm",
        lambda limit, selection_mode: [{"name": "AlphaClose", "code": "close"}],
    )
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": ""},
    )

    with pytest.raises(ValueError, match="LLM 未配置 API Key"):
        service.select_factors(
            prompt="寻找趋势突破因子",
            start_date="2024-01-01",
            end_date="2024-12-31",
            universe="single_stock",
            benchmark="hs300",
        )


def test_manual_genetic_candidates_exclude_rdagent_and_auto_mining_outputs(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.services.factor_service.factor_service.get_all_factors",
        lambda: [
            {
                "name": "ManualAlpha",
                "code": "close / open",
                "scope_type": "stock",
                "origin_type": "manual",
                "target_universe": "",
                "target_stock_code": "",
                "is_active": True,
                "category": "手工因子",
                "task_metadata": {},
            },
            {
                "name": "RejectedRDAgentAlpha",
                "code": "ts_rank(close, 5)",
                "scope_type": "stock",
                "origin_type": "rdagent_mining",
                "target_universe": "",
                "target_stock_code": "",
                "is_active": True,
                "category": "RDAgent 挖掘",
                "task_metadata": {"review_status": "pending"},
            },
            {
                "name": "AutoMiningAlpha",
                "code": "rank(ts_mean(close, 5))",
                "scope_type": "stock",
                "origin_type": "auto_mining",
                "target_universe": "",
                "target_stock_code": "",
                "is_active": True,
                "category": "自动挖掘",
                "task_metadata": {},
            },
        ],
    )

    candidates = factor_selection_service.load_factor_candidates_for_llm(
        limit=10,
        selection_mode="manual_genetic",
    )

    assert [item["name"] for item in candidates] == ["ManualAlpha"]


def test_select_factors_normalizes_manual_stability_direction_and_adds_hard_constraints(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.factor_selection_service.load_factor_candidates_for_llm",
        lambda limit, selection_mode: [
            {"name": "AlphaClose", "code": "close", "category": "price", "description": "close factor"},
        ],
    )
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
    )

    def fake_build_prompt(request, candidates):
        captured["direction"] = request.direction
        captured["selection_mode"] = request.selection_mode
        return "LLM PROMPT"

    def fake_select_with_llm(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return {
            "selected_factors": ["AlphaClose"],
            "selection_rationale": "选择基础价格因子。",
            "per_factor_reason": {"AlphaClose": "收盘价可直接作为单股票时间序列输入。"},
        }

    monkeypatch.setattr("backend.services.auto_factor_mining_service.build_llm_factor_selector_prompt", fake_build_prompt)
    monkeypatch.setattr(service, "_select_factors_with_llm", fake_select_with_llm)

    result = service.select_factors(
        prompt="优先稳定性和可解释性",
        direction="stability",
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="single_stock",
        benchmark="hs300",
        max_factor_count=1,
        candidate_limit=80,
        selection_mode="manual_genetic",
    )

    assert captured["direction"] == "report_sharpe"
    assert captured["selection_mode"] == "manual_genetic"
    assert "手动遗传挖掘硬约束" in captured["prompt"]
    assert "不要选择横截面、股票池依赖、目标股票绑定" in captured["prompt"]
    assert result["selected_factors"] == ["AlphaClose"]
    assert result["llm_used"] is True
    assert result["llm_model"] == "deepseek-chat"
    assert result["candidate_count"] == 1


def test_select_factors_with_llm_filters_unknown_names(monkeypatch) -> None:
    service = AutoFactorMiningService()

    def fake_run_async_tool(coro):
        coro.close()
        return {
            "content": {
                "selected_factors": ["GhostFactor", "AlphaClose", "AlphaClose"],
                "selection_rationale": "已选择有效因子",
                "per_factor_reason": {"AlphaClose": "价格因子有效"},
            },
            "response_id": "resp-unknown-filter",
            "provider": "openai_compatible",
        }

    monkeypatch.setattr(service, "_run_async_tool", fake_run_async_tool)

    result = service._select_factors_with_llm(
        prompt="LLM PROMPT",
        max_factor_count=3,
        candidates=[
            {"name": "AlphaClose", "code": "close"},
            {"name": "AlphaVolume", "code": "volume"},
        ],
        llm_config={"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
    )

    assert result["selected_factors"] == ["AlphaClose"]
    assert result["per_factor_reason"] == {"AlphaClose": "价格因子有效"}
    assert result["llm_response_id"] == "resp-unknown-filter"
    assert result["llm_provider"] == "openai_compatible"


def test_select_factors_with_llm_requests_json_object_response(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    class FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured["client_kwargs"] = kwargs

        @property
        def chat(self):
            def fake_create(**kwargs):
                captured["create_kwargs"] = kwargs
                return FakeResponse(
                    '{"selected_factors":["AlphaClose"],"selection_rationale":"已选择有效因子","per_factor_reason":{"AlphaClose":"价格因子有效"}}'
                )

            return SimpleNamespace(
                completions=SimpleNamespace(
                    create=fake_create
                )
            )

    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=FakeClient))

    service._select_factors_with_llm(
        prompt="LLM PROMPT",
        max_factor_count=2,
        candidates=[
            {"name": "AlphaClose", "code": "close"},
            {"name": "AlphaVolume", "code": "volume"},
        ],
        llm_config={"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
    )

    assert captured["client_kwargs"] == {"api_key": "test-key", "base_url": "https://example.com/v1"}
    assert captured["create_kwargs"]["response_format"] == {"type": "json_object"}


def test_select_factors_with_llm_supports_structured_message_content(monkeypatch) -> None:
    service = AutoFactorMiningService()

    class FakeResponse:
        def __init__(self) -> None:
            self.id = "resp-structured"
            self.choices = [
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=[
                            {"type": "text", "text": '{"selected_factors":["AlphaVolume"],'},
                            {"type": "text", "text": '"selection_rationale":"结构化返回也能解析",'},
                            {"type": "text", "text": '"per_factor_reason":{"AlphaVolume":"量能因子有效"}}'},
                        ]
                    )
                )
            ]

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        @property
        def chat(self):
            return SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: FakeResponse()
                )
            )

    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=FakeClient))

    result = service._select_factors_with_llm(
        prompt="LLM PROMPT",
        max_factor_count=2,
        candidates=[
            {"name": "AlphaClose", "code": "close"},
            {"name": "AlphaVolume", "code": "volume"},
        ],
        llm_config={"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
    )

    assert result["selected_factors"] == ["AlphaVolume"]
    assert result["selection_rationale"] == "结构化返回也能解析"
    assert result["per_factor_reason"] == {"AlphaVolume": "量能因子有效"}
    assert result["llm_response_id"] == "resp-structured"


def test_write_candidate_report_uses_reference_generator(monkeypatch, tmp_path) -> None:
    service = AutoFactorMiningService()
    calls: dict[str, object] = {}

    def fake_generate_report(
        ls_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
        title: str = "",
        output_dir: str | None = None,
        periods_per_year: int = 252,
    ) -> dict:
        calls["ls_returns"] = ls_returns
        calls["benchmark_returns"] = benchmark_returns
        calls["title"] = title
        calls["output_dir"] = output_dir
        calls["periods_per_year"] = periods_per_year
        report_path = tmp_path / "backtest_report_20260607_181500.html"
        report_path.write_text("<html>reference report</html>", encoding="utf-8")
        return {
            "report_path": str(report_path),
            "metrics": {"sharpe": 1.23, "cagr": 0.18, "max_drawdown": 0.12},
        }

    monkeypatch.setattr("backend.services.auto_factor_mining_service.generate_report", fake_generate_report)

    strategy_returns = pd.Series(
        [0.01, -0.02, 0.03],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        dtype=float,
    )
    benchmark_returns = pd.Series(
        [0.001, 0.0, 0.002],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        dtype=float,
    )

    metrics, report_url = service._write_candidate_report(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        periods_per_year=50,
    )

    assert calls["ls_returns"] is strategy_returns
    assert calls["benchmark_returns"] is benchmark_returns


def test_evaluate_expression_prefers_quantgpt_panel_execution(monkeypatch) -> None:
    service = AutoFactorMiningService()

    trade_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    stock_codes = [f"S{i:02d}" for i in range(8)]
    panel_rows = []
    factor_values = []
    for date_index, trade_date in enumerate(trade_dates):
        for stock_index, stock_code in enumerate(stock_codes):
            panel_rows.append(
                {
                    "trade_date": trade_date,
                    "stock_code": stock_code,
                    "close": 10 + stock_index + date_index,
                }
            )
            factor_values.append(float(stock_index))

    panel_df = pd.DataFrame(
        panel_rows
    )
    factor_series = pd.Series(factor_values, index=panel_df.index, dtype=float)

    monkeypatch.setattr(
        service._quantgpt_engine,
        "build_panel_data",
        lambda **kwargs: panel_df.copy(),
    )
    monkeypatch.setattr(
        service,
        "_validate_candidate_expression",
        lambda **kwargs: {
            "success": True,
            "valid": True,
            "message": "OK",
            "raw": {
                "input_expression": kwargs["expression"],
                "adapted_expression": "rank(close)",
            },
        },
    )
    monkeypatch.setattr(
        service._quantgpt_engine,
        "execute_on_panel",
        lambda panel, expression: SimpleNamespace(
            factor_series=factor_series,
            engine_type="quantgpt",
            dialect="quantgpt_local",
            canonical_expression="rank(close)",
            canonical_ast={"operators": ["rank"], "fields": ["close"]},
            diagnostics=[],
            execution_meta={"panel_rows": len(panel)},
            metrics_source="quantgpt_expression_engine",
        ),
    )
    monkeypatch.setattr(
        service,
        "_load_benchmark_returns",
        lambda benchmark, start_date, end_date: pd.Series(dtype=float),
    )
    monkeypatch.setattr(
        service,
        "_write_candidate_report",
        lambda strategy_returns, benchmark_returns, periods_per_year: ({"sharpe": 1.0}, "/reports/mock.html"),
    )

    result = service.evaluate_expression(
        expression="rank(close)",
        prompt="test quantgpt path",
        stock_codes=stock_codes,
        start_date="2024-01-01",
        end_date="2024-01-31",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        direction="score",
        neutralize_industry=False,
        neutralize_cap=False,
    )

    assert result is not None
    assert result.engine_type == "quantgpt"
    assert result.dialect == "quantgpt_local"
    assert result.canonical_expression == "rank(close)"
    assert result.execution_meta["panel_rows"] == len(panel_df)
    assert result.execution_meta["research_tools"]["validation"]["valid"] is True


def test_evaluate_expression_returns_none_when_quantgpt_fails(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        service,
        "_validate_candidate_expression",
        lambda **kwargs: {"success": True, "valid": True, "message": "OK", "raw": {}},
    )
    monkeypatch.setattr(
        service,
        "_diagnose_candidate_failure",
        lambda **kwargs: {
            "success": True,
            "report": "表达式需要替换字段",
            "key_findings": ["字段不支持"],
            "improvement_suggestions": ["改用 close"],
            "raw": {},
        },
    )
    monkeypatch.setattr(
        service._quantgpt_engine,
        "build_panel_data",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("quantgpt failed")),
    )

    result = service.evaluate_expression(
        expression="close",
        prompt="test invalid path",
        stock_codes=["S001"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        direction="score",
        neutralize_industry=False,
        neutralize_cap=False,
    )

    assert result is None


def test_evaluate_expression_returns_none_when_quantgpt_panel_is_empty(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        service,
        "_validate_candidate_expression",
        lambda **kwargs: {"success": True, "valid": True, "message": "OK", "raw": {}},
    )
    monkeypatch.setattr(
        service,
        "_diagnose_candidate_failure",
        lambda **kwargs: {
            "success": True,
            "report": "panel 为空",
            "key_findings": ["无有效 panel"],
            "improvement_suggestions": ["调整 universe"],
            "raw": {},
        },
    )
    monkeypatch.setattr(
        service._quantgpt_engine,
        "build_panel_data",
        lambda **kwargs: pd.DataFrame(),
    )

    result = service.evaluate_expression(
        expression="close",
        prompt="test empty panel invalid",
        stock_codes=["S001"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        direction="score",
        neutralize_industry=False,
        neutralize_cap=False,
    )

    assert result is None


def test_evaluate_expression_returns_none_when_quantgpt_validation_fails(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        service,
        "_validate_candidate_expression",
        lambda **kwargs: {
            "success": True,
            "valid": False,
            "message": "syntax error",
            "raw": {"adapted_expression": "bad(expr)"},
        },
    )
    monkeypatch.setattr(
        service._quantgpt_engine,
        "build_panel_data",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("validation fail should short-circuit")),
    )

    result = service.evaluate_expression(
        expression="bad(expr)",
        prompt="test validation fail",
        stock_codes=["S001"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        direction="score",
        neutralize_industry=False,
        neutralize_cap=False,
    )

    assert result is None


def test_quantgpt_engine_adapts_generator_dialect_before_execution() -> None:
    service = AutoFactorMiningService()

    frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "stock_code": ["S001"] * 6,
            "close": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0],
            "volume": [100.0, 120.0, 140.0, 160.0, 180.0, 200.0],
        }
    )

    result = service._quantgpt_engine.execute_on_panel(frame, "Ref($close, 1)")

    assert result.factor_series is not None
    assert result.canonical_expression == "ts_shift(close,1)"
    assert float(result.factor_series.iloc[2]) == 11.0


def test_quantgpt_engine_builds_panel_from_unnamed_datetime_index() -> None:
    service = AutoFactorMiningService()

    raw_frame = pd.DataFrame(
        {
            "close": [10.0, 11.0, 12.0],
            "volume": [100.0, 120.0, 140.0],
            "amount": [1000.0, 1320.0, 1680.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    panel = service._quantgpt_engine.build_panel_data(
        stock_codes=["S001"],
        start_date="2024-01-01",
        end_date="2024-01-03",
        expression="rank(close)",
        stock_data_loader=lambda stock_code, start_date, end_date: raw_frame.copy(),
    )

    assert not panel.empty
    assert "trade_date" in panel.columns
    assert panel["trade_date"].notna().all()


def test_filter_supported_expressions_uses_quantgpt_engine_only(monkeypatch) -> None:
    service = AutoFactorMiningService()
    sample_frames = [
        pd.DataFrame(
            {
                "open": [10.0, 10.2, 10.4],
                "high": [10.3, 10.5, 10.7],
                "low": [9.8, 10.0, 10.2],
                "close": [10.1, 10.3, 10.6],
                "volume": [1000, 1100, 1200],
                "amount": [10000, 11330, 12720],
            }
        )
    ]

    monkeypatch.setattr(
        service._quantgpt_engine,
        "can_execute_on_frames",
        lambda expression, frames: expression == "rank(close)",
    )
    monkeypatch.setattr(
        factor_service.calculator,
        "calculate",
        lambda df, expr: (_ for _ in ()).throw(AssertionError("should not call factorhub calculator")),
    )

    supported = service._filter_supported_expressions(
        ["rank(close)", "bad(expr)"],
        sample_frames=sample_frames,
        limit=2,
    )

    assert supported == ["rank(close)"]


def test_run_auto_mining_raises_when_llm_candidates_are_not_executable(monkeypatch) -> None:
    service = AutoFactorMiningService()

    sample_df = pd.DataFrame(
        {
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000.0 + i * 10 for i in range(40)],
            "amount": [10000.0 + i * 100 for i in range(40)],
        }
    )

    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda base_factors: ["close", "volume"])
    monkeypatch.setattr(
        service,
        "generate_candidate_expressions",
        lambda **kwargs: ["rank(ts_mean(close, 5))", "ts_rank(correlation(close, volume, 10), 20)"],
    )
    monkeypatch.setattr(
        service,
        "_filter_supported_expressions",
        lambda expressions, *, sample_frames, limit: [],
    )

    service._data_service = SimpleNamespace(
        get_stock_universe=lambda universe, date: ["000001.SZ"],
        get_stock_data=lambda stock_code, start_date, end_date: sample_df.copy(),
    )
    try:
        service.run_auto_mining(
            prompt="提升量价复合因子的稳定性",
            base_factors=["close", "volume"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            universe="hs300",
            benchmark="hs300",
            n_groups=5,
            holding_period=5,
            n_candidates=1,
        )
    except ValueError as exc:
        assert str(exc) == "未生成可执行候选表达式"
    else:
        raise AssertionError("expected auto mining to fail when no QuantGPT candidate is executable")


def test_generate_candidate_expressions_falls_back_to_local_templates(monkeypatch) -> None:
    service = AutoFactorMiningService()

    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": ""},
    )

    expressions = service.generate_candidate_expressions(
        prompt="提升稳定性",
        base_factor_codes=["close", "volume"],
        n_candidates=2,
        previous_expressions=[],
        continuation_context={
            "primary_problem": "整体指标较均衡，但仍可围绕目标继续精修。",
            "suggested_actions": ["继续在当前因子族附近做结构性微调，优先保留高 rankIC 结构。"],
            "hold_reason": "上一轮没有暴露出足够明确的结构性短板，优先保持当前基础因子组合，先在表达式结构上做微调。",
            "metric_snapshot": {"turnover": 0.48, "rank_ic": 0.026},
        },
    )

    assert len(expressions) == 2


def test_fallback_templates_for_rankic_problem_prefer_interaction_signals() -> None:
    service = AutoFactorMiningService()

    expressions = service._build_fallback_candidate_expressions(
        base_factor_codes=["close", "volume"],
        n_candidates=3,
        previous_expressions=[],
        continuation_context={
            "primary_problem": "横截面 rankIC 偏弱，说明选股区分度不足。",
            "suggested_actions": ["优先补充新的量价交互或横截面离散度信息。"],
            "metric_snapshot": {"rank_ic": 0.011, "turnover": 0.42},
        },
    )

    assert any("ts_corr(close, volume, 10)" in expression for expression in expressions)
    assert any("ts_std" in expression for expression in expressions)


def test_fallback_templates_use_secondary_problem_as_constraint() -> None:
    service = AutoFactorMiningService()

    expressions = service._build_fallback_candidate_expressions(
        base_factor_codes=["close", "volume"],
        n_candidates=4,
        previous_expressions=[],
        continuation_context={
            "primary_problem": "横截面 rankIC 偏弱，说明选股区分度不足。",
            "secondary_problem": "L/S Sharpe 偏低，收益质量仍需改善。",
            "suggested_actions": [
                "下一轮只围绕“ls_sharpe”做小步优化，不要同时追求多个目标。",
                "优先保留当前有效结构，只调整与主要短板直接相关的基础因子或表达式局部。",
            ],
            "metric_snapshot": {"rank_ic": 0.011, "turnover": 0.44, "ls_sharpe": 0.58},
        },
    )

    assert expressions
    assert expressions[0] == "rank(ts_mean(ts_corr(close, volume, 10), 5))"
    assert all(expression != "rank(ts_delta(close, 5))" for expression in expressions[:3])


def test_fallback_templates_prioritize_newly_added_base_factor_codes() -> None:
    service = AutoFactorMiningService()

    expressions = service._build_fallback_candidate_expressions(
        base_factor_codes=["close/open", "(high-low)/open", "force_index_ma", "volatility_10"],
        n_candidates=4,
        previous_expressions=[],
        continuation_context={
            "primary_problem": "横截面 rankIC 偏弱，说明选股区分度不足。",
            "secondary_problem": "L/S Sharpe 偏低，收益质量仍需改善。",
            "base_factors": ["close_open_ratio", "high_low_ratio", "force_index_ma", "volatility_10"],
            "previous_base_factors": ["close_open_ratio", "high_low_ratio"],
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
            "suggested_actions": ["优先把新增量价或波动因子真正用进下一轮表达式。"],
            "metric_snapshot": {"rank_ic": 0.011, "turnover": 0.44, "ls_sharpe": 0.58},
        },
    )

    assert expressions
    assert any("force_index_ma" in expression or "volatility_10" in expression for expression in expressions[:2])


def test_fallback_templates_include_parent_expression_local_rewrites() -> None:
    service = AutoFactorMiningService()

    expressions = service._build_fallback_candidate_expressions(
        base_factor_codes=["close/open", "(high-low)/open", "force_index_ma"],
        n_candidates=4,
        previous_expressions=[],
        continuation_context={
            "primary_problem": "Sharpe 偏低，收益质量仍需改善。",
            "secondary_problem": "换手率偏高。",
            "parent_expression": "rank(ts_mean(close/open,5))",
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
            "suggested_actions": ["优先保留当前有效结构，只做局部调整。"],
            "metric_snapshot": {"turnover": 0.58, "ls_sharpe": 0.56},
        },
    )

    assert expressions
    assert expressions[0] == "rank(ts_mean(rank(ts_mean(close/open,5)), 3))"
    assert any("rank(ts_mean(rank(ts_mean(close/open,5)), 5) * ts_mean(force_index_ma, 5))" == expression for expression in expressions)


def test_fallback_templates_prioritize_parent_new_factor_interactions_when_adjusting_base_factors() -> None:
    service = AutoFactorMiningService()

    expressions = service._build_fallback_candidate_expressions(
        base_factor_codes=["close/open", "(high-low)/open", "force_index_ma", "volatility_10"],
        n_candidates=6,
        previous_expressions=[],
        continuation_context={
            "primary_problem": "横截面 rankIC 偏弱，说明选股区分度不足。",
            "secondary_problem": "L/S Sharpe 偏低，收益质量仍需改善。",
            "parent_expression": "rank(ts_mean(close/open,5))",
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
            "should_adjust_base_factors": True,
            "suggested_actions": ["优先把新增量价或波动因子真正用进下一轮表达式。"],
            "metric_snapshot": {"rank_ic": 0.011, "turnover": 0.44, "ls_sharpe": 0.58},
        },
    )

    assert expressions
    assert "force_index_ma" in expressions[0] or "volatility_10" in expressions[0]
    assert all(
        expression not in {
            "rank(ts_mean(rank(ts_mean(close/open,5)), 3))",
            "rank(ts_mean(rank(ts_mean(close/open,5)), 5))",
            "rank(ts_zscore(rank(ts_mean(close/open,5)), 20))",
        }
        for expression in expressions[:4]
    )


def test_score_evaluation_for_continuation_rewards_new_factor_usage_and_penalizes_high_turnover() -> None:
    service = AutoFactorMiningService()
    continuation_context = {
        "primary_problem": "横截面 rankIC 偏弱，说明选股区分度不足。",
        "secondary_problem": "L/S Sharpe 偏低，收益质量仍需改善。",
        "suggested_actions": ["优先把新增量价或波动因子真正用进下一轮表达式，并避免高换手。"],
        "previous_base_factor_codes": ["close/open", "(high-low)/open"],
        "parent_expression": "rank(ts_mean(close/open,5))",
    }
    base_factor_codes = ["close/open", "(high-low)/open", "force_index_ma", "volatility_change"]

    old_factor_eval = SimpleNamespace(
        score=20.0,
        expression="rank(ts_mean(close/open,5))",
        backtest_summary={"turnover": 0.82, "rank_ic_mean": 0.002},
    )
    new_factor_eval = SimpleNamespace(
        score=19.4,
        expression="rank(ts_mean(force_index_ma,5)/(1+ts_std(volatility_change,20)))",
        backtest_summary={"turnover": 0.43, "rank_ic_mean": 0.006},
    )

    old_rank = service._score_evaluation_for_continuation(
        evaluation=old_factor_eval,
        continuation_context=continuation_context,
        base_factor_codes=base_factor_codes,
    )
    new_rank = service._score_evaluation_for_continuation(
        evaluation=new_factor_eval,
        continuation_context=continuation_context,
        base_factor_codes=base_factor_codes,
    )

    assert new_rank > old_rank


def test_score_evaluation_for_continuation_prefers_parent_structure_plus_new_factor() -> None:
    service = AutoFactorMiningService()
    continuation_context = {
        "primary_problem": "Sharpe 偏低，收益质量仍需改善。",
        "secondary_problem": "换手率偏高。",
        "suggested_actions": ["优先保留当前有效结构，只引入新增波动或量价因子。"],
        "previous_base_factor_codes": ["close/open", "(high-low)/open"],
        "parent_expression": "rank(ts_mean(close/open,5))",
    }
    base_factor_codes = ["close/open", "(high-low)/open", "force_index_ma"]

    parent_anchored_eval = SimpleNamespace(
        score=18.6,
        expression="rank(ts_mean(rank(ts_mean(close/open,5)), 5) * ts_mean(force_index_ma, 5))",
        backtest_summary={"turnover": 0.46, "rank_ic_mean": 0.005},
    )
    fully_rewritten_eval = SimpleNamespace(
        score=18.9,
        expression="rank(ts_corr(force_index_ma, (high-low)/open, 10))",
        backtest_summary={"turnover": 0.46, "rank_ic_mean": 0.005},
    )

    parent_rank = service._score_evaluation_for_continuation(
        evaluation=parent_anchored_eval,
        continuation_context=continuation_context,
        base_factor_codes=base_factor_codes,
    )
    rewritten_rank = service._score_evaluation_for_continuation(
        evaluation=fully_rewritten_eval,
        continuation_context=continuation_context,
        base_factor_codes=base_factor_codes,
    )

    assert parent_rank > rewritten_rank


def test_continuation_parent_rejects_round_when_best_expression_skips_expected_new_factors() -> None:
    service = AutoFactorMiningService()

    feedback = service._build_continuation_feedback(
        previous_best_score=39.02,
        current_best_score=42.9,
        retention_count=1,
        direction="ls_sharpe",
        previous_best_result={
            "factors": [
                {
                    "score": 39.02,
                    "backtest_summary": {
                        "rank_ic_mean": 0.003116,
                        "long_short_sharpe": 1.0289,
                        "long_short_annual": 0.1702,
                        "turnover": 0.272801,
                    },
                    "report_metrics": {"sharpe": 0.845355},
                }
            ]
        },
        current_result={
            "factors": [
                {
                    "score": 42.9,
                    "backtest_summary": {
                        "rank_ic_mean": 0.007687,
                        "long_short_sharpe": 1.0504,
                        "long_short_annual": 0.1679,
                        "turnover": 0.208737,
                    },
                    "report_metrics": {"sharpe": 0.860457},
                }
            ]
        },
        factor_usage={
            "used_base_factors": ["high_low_ratio"],
            "unused_base_factors": ["close_open_ratio", "obv_slope", "force_index_ma"],
            "used_new_factors": [],
            "unused_new_factors": ["obv_slope", "force_index_ma"],
        },
        continuation_hypothesis={
            "selected_for_next_round": ["obv_slope", "force_index_ma"],
            "should_adjust_base_factors": True,
        },
    )

    assert feedback["accepted_as_best"] is False
    assert feedback["fallback_parent_strategy"] == "best_score_so_far"
    assert "未吸收新增基础因子" in feedback["reason"]


def test_prioritize_supported_expressions_for_continuation_prefers_new_factor_usage() -> None:
    service = AutoFactorMiningService()

    prioritized = service._prioritize_supported_expressions_for_continuation(
        expressions=[
            "rank(ts_mean(close/open,5))",
            "rank(ts_mean(force_index_ma,5)/(1+ts_std(volatility_change,20)))",
            "rank(ts_mean((high-low)/open,10))",
        ],
        continuation_context={
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
        },
        base_factor_codes=["close/open", "(high-low)/open", "force_index_ma", "volatility_change"],
    )

    assert prioritized[0] == "rank(ts_mean(force_index_ma,5)/(1+ts_std(volatility_change,20)))"


def test_prioritize_supported_expressions_for_continuation_prefers_parent_anchored_small_step() -> None:
    service = AutoFactorMiningService()

    prioritized = service._prioritize_supported_expressions_for_continuation(
        expressions=[
            "rank(ts_corr(force_index_ma, (high-low)/open, 10))",
            "rank(ts_mean(rank(ts_mean(close/open,5)), 5) * ts_mean(force_index_ma, 5))",
            "rank(ts_mean(close/open,5))",
        ],
        continuation_context={
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
            "parent_expression": "rank(ts_mean(close/open,5))",
        },
        base_factor_codes=["close/open", "(high-low)/open", "force_index_ma"],
    )

    assert prioritized[0] == "rank(ts_mean(rank(ts_mean(close/open,5)), 5) * ts_mean(force_index_ma, 5))"


def test_prioritize_supported_expressions_for_continuation_penalizes_pure_parent_smoothing() -> None:
    service = AutoFactorMiningService()

    prioritized = service._prioritize_supported_expressions_for_continuation(
        expressions=[
            "rank(ts_mean(rank(ts_mean(close/open,5)), 3))",
            "rank(ts_mean(rank(ts_mean(close/open,5)), 5) * ts_mean(force_index_ma, 5))",
            "rank(ts_corr(rank(ts_mean(close/open,5)), force_index_ma, 10))",
        ],
        continuation_context={
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
            "parent_expression": "rank(ts_mean(close/open,5))",
            "should_adjust_base_factors": True,
        },
        base_factor_codes=["close/open", "(high-low)/open", "force_index_ma"],
    )

    assert prioritized[-1] == "rank(ts_mean(rank(ts_mean(close/open,5)), 3))"


def test_select_best_evaluation_for_continuation_promotes_new_factor_candidate_when_gap_is_small() -> None:
    service = AutoFactorMiningService()

    top_eval = SimpleNamespace(
        score=42.9,
        expression="rank(ts_mean(ts_mean(((high-low)/open)*rank(volume/ts_mean(volume,10)),10),3))",
    )
    new_factor_eval = SimpleNamespace(
        score=39.8,
        expression="ts_mean(((high-low)/open)*ts_mean(volume/ts_mean(volume,20),5),3)*ts_rank(ts_mean((close-ts_shift(close,1))*volume,timeperiod=13),10)",
    )
    fallback_eval = SimpleNamespace(
        score=18.0,
        expression="rank(ts_mean(close/open,5))",
    )

    reordered = service._select_best_evaluation_for_continuation(
        evaluations=[top_eval, new_factor_eval, fallback_eval],
        continuation_context={
            "should_adjust_base_factors": True,
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
        },
        base_factor_codes=["close/open", "(high-low)/open", "SMA((close - close.shift(1)) * volume, timeperiod=13)"],
    )

    assert reordered[0] is new_factor_eval


def test_select_best_evaluation_for_continuation_keeps_top_when_new_factor_gap_is_too_large() -> None:
    service = AutoFactorMiningService()

    top_eval = SimpleNamespace(
        score=42.9,
        expression="rank(ts_mean(ts_mean(((high-low)/open)*rank(volume/ts_mean(volume,10)),10),3))",
    )
    new_factor_eval = SimpleNamespace(
        score=26.0,
        expression="ts_mean(((high-low)/open)*ts_mean(volume/ts_mean(volume,20),5),3)*ts_rank(ts_mean((close-ts_shift(close,1))*volume,timeperiod=13),10)",
    )

    reordered = service._select_best_evaluation_for_continuation(
        evaluations=[top_eval, new_factor_eval],
        continuation_context={
            "should_adjust_base_factors": True,
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
        },
        base_factor_codes=["close/open", "(high-low)/open", "SMA((close - close.shift(1)) * volume, timeperiod=13)"],
    )

    assert reordered[0] is top_eval


def test_select_best_evaluation_for_continuation_promotes_new_factor_when_target_metric_is_better() -> None:
    service = AutoFactorMiningService()

    top_eval = SimpleNamespace(
        score=75.15,
        expression="rank(ts_zscore(rank(ts_mean(ts_delta(close/open,5),5)),20))",
        backtest_summary={"long_short_sharpe": 1.1435},
        report_metrics={},
        wq_brain={},
    )
    new_factor_eval = SimpleNamespace(
        score=39.32,
        expression="rank(ts_mean(ts_mean((close-ts_shift(close,1))*volume,timeperiod=13),8)/(1+ts_mean((high-low)/open,10)))",
        backtest_summary={"long_short_sharpe": -0.5981},
        report_metrics={},
        wq_brain={},
    )
    better_metric_new_factor_eval = SimpleNamespace(
        score=57.0,
        expression="rank(ts_mean(force_index_ma,8)/(1+ts_mean((high-low)/open,10)))",
        backtest_summary={"long_short_sharpe": 1.34},
        report_metrics={},
        wq_brain={},
    )

    reordered = service._select_best_evaluation_for_continuation(
        evaluations=[top_eval, new_factor_eval, better_metric_new_factor_eval],
        continuation_context={
            "should_adjust_base_factors": True,
            "recommended_goal": "ls_sharpe",
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
        },
        base_factor_codes=["close/open", "(high-low)/open", "force_index_ma"],
    )

    assert reordered[0] is better_metric_new_factor_eval


def test_generate_candidates_with_llm_prompt_lists_only_new_base_factor_codes(monkeypatch) -> None:
    service = AutoFactorMiningService()
    captured: dict[str, object] = {}

    class FakeResponse:
        def __init__(self) -> None:
            self.choices = [SimpleNamespace(message=SimpleNamespace(content='["rank(force_index_ma)"]'))]

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured["client_kwargs"] = kwargs

        @property
        def chat(self):
            return SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (captured.setdefault("request_kwargs", kwargs), FakeResponse())[1]
                )
            )

    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=FakeClient))

    service._generate_candidates_with_llm(
        prompt="继续优化",
        base_factor_codes=["close/open", "(high-low)/open", "force_index_ma", "volatility_change"],
        n_candidates=2,
        llm_config={"api_key": "test-key", "model": "deepseek-chat", "base_url": "https://example.com/v1"},
        previous_expressions=[],
        continuation_context={
            "primary_problem": "横截面 rankIC 偏弱",
            "secondary_problem": "L/S Sharpe 偏低",
            "previous_base_factors": ["close_open_ratio", "high_low_ratio"],
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
            "recommended_goal": "ls_sharpe",
            "suggested_actions": ["优先使用新增因子"],
            "metric_snapshot": {"rank_ic": 0.01},
        },
    )

    user_prompt = captured["request_kwargs"]["messages"][1]["content"]
    assert "本轮新增基础因子表达式" in user_prompt
    assert '"force_index_ma"' in user_prompt
    assert '"volatility_change"' in user_prompt
    assert '"close/open"' not in user_prompt.split("本轮新增基础因子表达式：", 1)[1].split("\n", 1)[0]


def test_quantgpt_engine_supports_resolved_continuation_factor_codes() -> None:
    service = AutoFactorMiningService()
    sample_frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "stock_code": ["000001.SZ"] * 60,
            "open": [10 + i * 0.1 for i in range(60)],
            "high": [10.5 + i * 0.1 for i in range(60)],
            "low": [9.5 + i * 0.1 for i in range(60)],
            "close": [10.2 + i * 0.1 for i in range(60)],
            "volume": [1000 + i * 10 for i in range(60)],
            "amount": [10000 + i * 200 for i in range(60)],
        }
    )

    resolved_codes = service.resolve_base_factor_codes(["force_index_ma", "volatility_10"])

    assert resolved_codes
    assert service._quantgpt_engine.can_execute_on_frames(resolved_codes[0], [sample_frame.copy()]) is True
    assert service._quantgpt_engine.can_execute_on_frames(resolved_codes[1], [sample_frame.copy()]) is True


def test_filter_supported_expressions_accepts_resolved_new_factor_codes_for_continuation() -> None:
    service = AutoFactorMiningService()
    sample_frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "stock_code": ["000001.SZ"] * 60,
            "open": [10 + i * 0.1 for i in range(60)],
            "high": [10.5 + i * 0.1 for i in range(60)],
            "low": [9.5 + i * 0.1 for i in range(60)],
            "close": [10.2 + i * 0.1 for i in range(60)],
            "volume": [1000 + i * 10 for i in range(60)],
            "amount": [10000 + i * 200 for i in range(60)],
        }
    )

    resolved_codes = service.resolve_base_factor_codes(
        ["close_open_ratio", "high_low_ratio", "force_index_ma", "volatility_10"]
    )
    expressions = service._build_fallback_candidate_expressions(
        base_factor_codes=resolved_codes,
        n_candidates=5,
        previous_expressions=[],
        continuation_context={
            "primary_problem": "横截面 rankIC 偏弱，说明选股区分度不足。",
            "secondary_problem": "L/S Sharpe 偏低，收益质量仍需改善。",
            "previous_base_factor_codes": service.resolve_base_factor_codes(["close_open_ratio", "high_low_ratio"]),
            "suggested_actions": ["优先把新增量价或波动因子真正用进下一轮表达式。"],
            "metric_snapshot": {"rank_ic": 0.011, "turnover": 0.44, "ls_sharpe": 0.58},
        },
    )

    supported = service._filter_supported_expressions(
        expressions,
        sample_frames=[sample_frame.copy()],
        limit=20,
    )

    assert supported


def test_build_required_continuation_expressions_prioritizes_new_factor_usage() -> None:
    service = AutoFactorMiningService()

    expressions = service._build_required_continuation_expressions(
        base_factor_codes=["close/open", "(high-low)/open", "force_index_ma", "downside_risk"],
        continuation_context={
            "previous_base_factor_codes": ["close/open", "(high-low)/open"],
            "parent_expression": "rank(ts_mean(close/open,5))",
            "should_adjust_base_factors": True,
        },
        limit=6,
    )

    assert expressions
    assert all("force_index_ma" in expression or "downside_risk" in expression for expression in expressions[:4])


def test_expression_adapter_makes_obv_slope_and_downside_risk_executable() -> None:
    service = AutoFactorMiningService()
    sample_frame = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=80, freq="D"),
            "stock_code": ["000001.SZ"] * 80,
            "open": [10 + i * 0.1 for i in range(80)],
            "high": [10.5 + i * 0.1 for i in range(80)],
            "low": [9.5 + i * 0.1 for i in range(80)],
            "close": [10.2 + i * 0.1 for i in range(80)],
            "volume": [1000 + i * 10 for i in range(80)],
            "amount": [10000 + i * 200 for i in range(80)],
        }
    )

    resolved_codes = service.resolve_base_factor_codes(["obv_slope", "downside_risk"])

    assert resolved_codes
    assert service._quantgpt_engine.can_execute_on_frames(resolved_codes[0], [sample_frame.copy()]) is True
    assert service._quantgpt_engine.can_execute_on_frames(resolved_codes[1], [sample_frame.copy()]) is True


def test_validation_service_bypasses_remote_when_local_mode_has_no_quantgpt() -> None:
    class FakeClient:
        def is_configured(self) -> bool:
            return False

        async def validate_expression(self, expression: str, mode: str = "local") -> str:
            raise AssertionError("should not call remote validation when QuantGPT is not configured")

    service = ValidationService(client=FakeClient())
    response = asyncio.run(service.validate_expression("rank(close/open)", "local"))

    assert response.success is True
    assert response.valid is True
    assert response.raw["bypassed_remote_validation"] is True


def test_build_round_evaluation_focuses_on_primary_problem_only() -> None:
    service = AutoFactorMiningService()

    evaluation = SimpleNamespace(
        score=22.22,
        report_metrics={"sharpe": 0.38, "max_drawdown": -0.19},
        backtest_summary={
            "long_short_sharpe": 0.59,
            "long_short_annual": 0.086,
            "rank_ic_mean": -0.0062,
            "turnover": 0.87,
            "wq_fitness": 0.18,
        },
        interpretation={
            "weaknesses": [
                "横截面 rankIC 偏弱，说明选股区分度不足。",
                "L/S Sharpe 偏低，收益质量仍需改善。",
                "换手率偏高，可能影响真实可交易性。",
            ],
            "next_steps": [
                "引入更强的排序或量价交互项，提升横截面区分度。",
                "优先降低噪音与波动暴露，减少极端反转信号。",
                "通过更平滑的基础因子或中周期信号压低换手。",
            ],
        },
        execution_meta={"research_tools": {"validation": {"valid": True}}},
    )

    result = service._build_round_evaluation(
        prompt="提升表现",
        base_factors=["close_open_ratio", "high_low_ratio"],
        best_evaluation=evaluation,
        direction="ls_sharpe",
    )

    assert result["primary_problem"] == "L/S Sharpe 偏低，收益质量仍需改善。"
    assert result["secondary_problem"] == "横截面 rankIC 偏弱，说明选股区分度不足。"
    assert result["recommended_goal"] == "ls_sharpe"
    assert result["suggested_actions"][0] == "下一轮只围绕“ls_sharpe”做小步优化，不要同时追求多个目标。"
    assert result["suggested_actions"][-1] == "先评估当前基础因子是否存在语义重复或风格过于单一，再决定补充或替换。"
    assert all("压低换手" not in action for action in result["suggested_actions"])


def test_build_round_evaluation_prioritizes_current_round_metric_regression_over_weakness_order() -> None:
    service = AutoFactorMiningService()

    evaluation = SimpleNamespace(
        score=10.38,
        report_metrics={"sharpe": 0.21, "max_drawdown": -0.18, "volatility": 0.22},
        backtest_summary={
            "long_short_sharpe": -0.12,
            "long_short_annual": 0.09,
            "rank_ic_mean": -0.0033,
            "turnover": 0.17,
            "wq_fitness": 0.11,
        },
        interpretation={
            "weaknesses": [
                "横截面 rankIC 偏弱，说明选股区分度不足。",
                "L/S Sharpe 偏低，收益质量仍需改善。",
                "换手率偏高，可能影响真实可交易性。",
            ],
            "next_steps": [
                "优先提高收益质量。",
            ],
        },
        execution_meta={"research_tools": {"validation": {"valid": True}}},
    )

    result = service._build_round_evaluation(
        prompt="继续提升表现",
        base_factors=["close_open_ratio", "high_low_ratio", "obv_slope"],
        best_evaluation=evaluation,
        direction="ls_sharpe",
    )

    assert result["primary_problem"] == "L/S Sharpe 偏低，收益质量仍需改善。"
    assert result["secondary_problem"] == "横截面 rankIC 偏弱，说明选股区分度不足。"
    assert result["metric_snapshot"]["ls_sharpe"] == -0.12
    assert result["metric_snapshot"]["rank_ic"] == -0.0033


def test_build_round_evaluation_uses_metric_specific_problem_text_for_negative_sharpe() -> None:
    service = AutoFactorMiningService()

    evaluation = SimpleNamespace(
        score=17.68,
        report_metrics={"sharpe": -1.12, "max_drawdown": -0.61, "volatility": 0.15},
        backtest_summary={
            "long_short_sharpe": -0.93,
            "long_short_annual": -0.1358,
            "rank_ic_mean": -0.0109,
            "turnover": 0.17,
            "wq_fitness": 0.83,
        },
        interpretation={
            "weaknesses": [
                "横截面 rankIC 偏弱，说明选股区分度不足。",
                "L/S Sharpe 偏低，收益质量仍需改善。",
            ],
            "next_steps": ["先恢复稳定性。"],
        },
        execution_meta={"research_tools": {"validation": {"valid": True}}},
    )

    result = service._build_round_evaluation(
        prompt="继续提升表现",
        base_factors=["close_open_ratio", "high_low_ratio", "volatility_10"],
        best_evaluation=evaluation,
        direction="ls_sharpe",
    )

    assert result["primary_problem"] == "L/S Sharpe 已转负（-0.93），收益质量明显恶化，当前结构需要先止损并恢复稳定性。"
    assert result["secondary_problem"] == "横截面 rankIC 偏弱，说明选股区分度不足。"


def test_build_continuation_context_adds_defensive_instructions_when_sharpe_turns_negative() -> None:
    service = AutoFactorMiningService()

    result = {
        "round_evaluation": {
            "base_factors": ["Alpha1", "Alpha2"],
            "primary_problem": "L/S Sharpe 已转负（-0.59），收益质量明显恶化，当前结构需要先止损并恢复稳定性。",
            "secondary_problem": "横截面 rankIC 偏弱，说明选股区分度不足。",
            "recommended_goal": "ls_sharpe",
            "suggested_actions": ["先恢复稳定性。"],
            "metric_snapshot": {
                "score": 15.09,
                "ls_sharpe": -0.5906,
                "ls_return": -0.0939,
                "rank_ic": -0.011059,
                "turnover": 0.214992,
                "report_max_drawdown": -0.510827,
            },
        },
        "factors": [
            {
                "score": 15.09,
                "report_metrics": {"sharpe": -0.775, "max_drawdown": -0.510827},
                "backtest_summary": {
                    "long_short_sharpe": -0.5906,
                    "long_short_annual": -0.0939,
                    "rank_ic_mean": -0.011059,
                    "turnover": 0.214992,
                    "wq_fitness": 0.3903,
                },
                "interpretation": {
                    "weaknesses": [
                        "横截面 rankIC 偏弱，说明选股区分度不足。",
                        "L/S Sharpe 偏低，收益质量仍需改善。",
                    ],
                    "next_steps": ["先恢复稳定性。"],
                },
                "expression": "rank(ts_mean(close/open,5))",
                "raw_expression": "rank(ts_mean(close/open,5))",
            }
        ],
    }

    context = service.build_continuation_context(
        result=result,
        request_payload={"base_factors": ["Alpha1", "Alpha2"]},
        prompt="继续优化",
        factor_update_mode="append",
        additional_factor_count=2,
    )

    assert "恢复正向收益和稳定性" in context["selection_instructions"]
    assert "防御性" in context["selection_instructions"]
    assert "downside" in context["preferred_keywords"]


def test_build_continuation_context_adds_direction_repair_instructions_when_rankic_turns_negative() -> None:
    service = AutoFactorMiningService()

    result = {
        "round_evaluation": {
            "base_factors": ["Alpha1", "Alpha2"],
            "primary_problem": "横截面 rankIC 为负（-0.0111），说明当前表达式的排序方向已经失真，选股区分度不足。",
            "secondary_problem": "L/S Sharpe 偏低，收益质量仍需改善。",
            "recommended_goal": "ls_sharpe",
            "suggested_actions": ["优先纠正排序方向。"],
            "metric_snapshot": {
                "score": 15.09,
                "ls_sharpe": -0.5906,
                "ls_return": -0.0939,
                "rank_ic": -0.011059,
                "turnover": 0.214992,
            },
        },
        "factors": [
            {
                "score": 15.09,
                "report_metrics": {"sharpe": -0.775, "max_drawdown": -0.510827},
                "backtest_summary": {
                    "long_short_sharpe": -0.5906,
                    "long_short_annual": -0.0939,
                    "rank_ic_mean": -0.011059,
                    "turnover": 0.214992,
                    "wq_fitness": 0.3903,
                },
                "interpretation": {
                    "weaknesses": [
                        "横截面 rankIC 偏弱，说明选股区分度不足。",
                        "L/S Sharpe 偏低，收益质量仍需改善。",
                    ],
                    "next_steps": ["优先纠正排序方向。"],
                },
                "expression": "rank(ts_mean(close/open,5))",
                "raw_expression": "rank(ts_mean(close/open,5))",
            }
        ],
    }

    context = service.build_continuation_context(
        result=result,
        request_payload={"base_factors": ["Alpha1", "Alpha2"]},
        prompt="继续优化",
        factor_update_mode="append",
        additional_factor_count=2,
    )

    assert "纠正排序方向" in context["selection_instructions"]
    assert "residual" in context["preferred_keywords"]
    assert "ratio" in context["avoid_keywords"]


def test_run_auto_mining_uses_fallback_templates_when_llm_returns_empty(monkeypatch) -> None:
    service = AutoFactorMiningService()

    sample_df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "stock_code": ["000001.SZ"] * 40,
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000.0 + i * 10 for i in range(40)],
            "amount": [10000.0 + i * 100 for i in range(40)],
        }
    )

    service._data_service = SimpleNamespace(
        get_stock_universe=lambda universe, date: ["000001.SZ"],
        get_stock_data=lambda stock_code, start_date, end_date: sample_df.copy(),
    )
    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda base_factors: ["close", "volume"])
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": ""},
    )
    monkeypatch.setattr(
        service,
        "_filter_supported_expressions",
        lambda expressions, *, sample_frames, limit: expressions[:limit],
    )

    def fake_evaluate_expression(**kwargs):
        expression = kwargs["expression"]
        return SimpleNamespace(
            expression=expression,
            score=77.0,
            grade="B",
            report_metrics={"sharpe": 1.1, "max_drawdown": 0.1},
            backtest_summary={
                "long_short_sharpe": 1.1,
                "long_short_annual": 0.18,
                "top_group_sharpe": 1.1,
                "monotonicity_score": 0.8,
                "spread": 0.03,
                "group_returns": {"top_minus_bottom_mean": 0.03},
                "rank_ic_mean": 0.05,
                "ic_mean": 0.05,
                "ic_ir": 1.0,
                "ic_win_rate": 0.7,
                "turnover": 0.2,
                "wq_fitness": 1.01,
            },
            wq_brain={
                "wq_rating": "B",
                "wq_fitness": 1.01,
                "wq_sharpe": 1.1,
                "wq_returns": 0.18,
                "wq_turnover": 0.2,
                "submittable": True,
            },
            component_scores={"total_score": 77.0},
            anti_overfit={"score": 75.0, "recommendation": "推荐", "tests": []},
            interpretation={
                "summary": "fallback ok",
                "weaknesses": [],
                "next_steps": ["继续迭代"],
                "rating": "B",
                "rating_reason": "推荐",
                "improvement_ideas": ["继续迭代"],
            },
            diagnostics=[],
            report_url="/api/mining/reports/fallback.html",
        )

    monkeypatch.setattr(service, "evaluate_expression", fake_evaluate_expression)

    result = service.run_auto_mining(
        prompt="提升量价复合因子的稳定性",
        base_factors=["close", "volume"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        n_candidates=1,
    )

    assert result["best_score"] == 77.0
    assert "rank(" in result["factors"][0]["expression"]


def test_run_auto_mining_uses_fallback_templates_when_llm_candidates_fail_precheck(monkeypatch) -> None:
    service = AutoFactorMiningService()

    sample_df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "stock_code": ["000001.SZ"] * 40,
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000.0 + i * 10 for i in range(40)],
            "amount": [10000.0 + i * 100 for i in range(40)],
        }
    )

    service._data_service = SimpleNamespace(
        get_stock_universe=lambda universe, date: ["000001.SZ"],
        get_stock_data=lambda stock_code, start_date, end_date: sample_df.copy(),
    )
    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda base_factors: ["close", "volume"])
    monkeypatch.setattr(
        "backend.services.auto_factor_mining_service.llm_config_service.get_runtime_config",
        lambda: {"api_key": "test-key"},
    )
    monkeypatch.setattr(
        service,
        "_generate_candidates_with_llm",
        lambda **kwargs: ["bad_expr_1"],
    )

    filter_calls = {"count": 0}

    def fake_filter(expressions, *, sample_frames, limit):
        filter_calls["count"] += 1
        if filter_calls["count"] == 1:
            return []
        return expressions[:limit]

    monkeypatch.setattr(service, "_filter_supported_expressions", fake_filter)

    def fake_evaluate_expression(**kwargs):
        expression = kwargs["expression"]
        return SimpleNamespace(
            expression=expression,
            score=79.0,
            grade="B",
            report_metrics={"sharpe": 1.0, "max_drawdown": 0.1},
            backtest_summary={
                "long_short_sharpe": 1.0,
                "long_short_annual": 0.16,
                "top_group_sharpe": 1.0,
                "monotonicity_score": 0.7,
                "spread": 0.02,
                "group_returns": {"top_minus_bottom_mean": 0.02},
                "rank_ic_mean": 0.04,
                "ic_mean": 0.04,
                "ic_ir": 0.9,
                "ic_win_rate": 0.65,
                "turnover": 0.3,
                "wq_fitness": 0.98,
            },
            wq_brain={
                "wq_rating": "B",
                "wq_fitness": 0.98,
                "wq_sharpe": 1.0,
                "wq_returns": 0.16,
                "wq_turnover": 0.3,
                "submittable": True,
            },
            component_scores={"total_score": 79.0},
            anti_overfit={"score": 75.0, "recommendation": "推荐", "tests": []},
            interpretation={
                "summary": "fallback after precheck ok",
                "weaknesses": [],
                "next_steps": ["继续迭代"],
                "rating": "B",
                "rating_reason": "推荐",
                "improvement_ideas": ["继续迭代"],
            },
            diagnostics=[],
            report_url="/api/mining/reports/fallback-precheck.html",
        )

    monkeypatch.setattr(service, "evaluate_expression", fake_evaluate_expression)

    result = service.run_auto_mining(
        prompt="提升量价复合因子的稳定性",
        base_factors=["close", "volume"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        n_candidates=1,
    )

    assert filter_calls["count"] >= 2
    assert result["best_score"] == 79.0
    assert "rank(" in result["factors"][0]["expression"]


def test_run_auto_mining_keeps_retrying_until_quantgpt_yields_valid_candidate(monkeypatch) -> None:
    service = AutoFactorMiningService()

    sample_df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "stock_code": ["000001.SZ"] * 40,
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000.0 + i * 10 for i in range(40)],
            "amount": [10000.0 + i * 100 for i in range(40)],
        }
    )

    service._data_service = SimpleNamespace(
        get_stock_universe=lambda universe, date: ["000001.SZ"],
        get_stock_data=lambda stock_code, start_date, end_date: sample_df.copy(),
    )
    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda base_factors: ["close", "volume"])

    generate_calls = {"count": 0}

    def fake_generate_candidate_expressions(**kwargs):
        generate_calls["count"] += 1
        if generate_calls["count"] == 1:
            return ["bad_expr_1"]
        return ["good_expr_2"]

    monkeypatch.setattr(service, "generate_candidate_expressions", fake_generate_candidate_expressions)
    monkeypatch.setattr(
        service,
        "_filter_supported_expressions",
        lambda expressions, *, sample_frames, limit: expressions[:limit],
    )
    monkeypatch.setattr(
        factor_service.calculator,
        "calculate",
        lambda df, expr: pd.Series(range(len(df)), index=df.index, dtype=float),
    )

    def fake_evaluate_expression(**kwargs):
        expression = kwargs["expression"]
        if expression == "bad_expr_1":
            return None
        return SimpleNamespace(
            expression=expression,
            score=88.0,
            grade="A",
            report_metrics={"sharpe": 1.2, "max_drawdown": 0.1},
            backtest_summary={
                "long_short_sharpe": 1.2,
                "long_short_annual": 0.2,
                "top_group_sharpe": 1.1,
                "monotonicity_score": 0.8,
                "spread": 0.03,
                "group_returns": {"top_minus_bottom_mean": 0.03},
                "rank_ic_mean": 0.05,
                "ic_mean": 0.05,
                "ic_ir": 1.0,
                "ic_win_rate": 0.7,
                "turnover": 0.2,
                "wq_fitness": 1.05,
            },
            wq_brain={
                "wq_rating": "A",
                "wq_fitness": 1.05,
                "wq_sharpe": 1.2,
                "wq_returns": 0.2,
                "wq_turnover": 0.2,
                "submittable": True,
            },
            component_scores={"total_score": 88.0},
            anti_overfit={"score": 80.0, "recommendation": "推荐", "tests": []},
            interpretation={
                "summary": "retry ok",
                "weaknesses": [],
                "next_steps": ["继续迭代"],
                "rating": "A",
                "rating_reason": "推荐",
                "improvement_ideas": ["继续迭代"],
            },
            diagnostics=[],
            report_url="/api/mining/reports/retry.html",
        )

    monkeypatch.setattr(service, "evaluate_expression", fake_evaluate_expression)

    result = service.run_auto_mining(
        prompt="提升量价复合因子的稳定性",
        base_factors=["close", "volume"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        universe="hs300",
        benchmark="hs300",
        n_groups=5,
        holding_period=5,
        n_candidates=1,
    )

    assert generate_calls["count"] >= 2
    assert result["best_score"] == 88.0
    assert result["factors"][0]["expression"] == "good_expr_2"


def test_run_auto_mining_accepts_bound_benchmark_and_report_helpers(monkeypatch) -> None:
    service = AutoFactorMiningService()

    trade_dates = pd.date_range("2024-01-01", periods=40, freq="D")
    stock_codes = [f"{i:06d}.SZ" for i in range(1, 9)]
    service._data_service = SimpleNamespace(
        get_stock_universe=lambda universe, date: stock_codes,
        get_stock_data=lambda stock_code, start_date, end_date: pd.DataFrame(
            {
                "stock_code": [stock_code] * 40,
                "open": [10.0 + i + (0.2 * stock_codes.index(stock_code)) for i in range(40)],
                "high": [10.5 + i + (0.2 * stock_codes.index(stock_code)) for i in range(40)],
                "low": [9.5 + i + (0.2 * stock_codes.index(stock_code)) for i in range(40)],
                "close": [10.2 + i + (0.2 * stock_codes.index(stock_code)) for i in range(40)],
                "volume": [1000.0 + i * 10 + (20 * stock_codes.index(stock_code)) for i in range(40)],
                "amount": [10000.0 + i * 100 + (200 * stock_codes.index(stock_code)) for i in range(40)],
            },
            index=trade_dates,
        ),
        get_benchmark_returns=lambda benchmark, start_date, end_date: pd.DataFrame(
            {
                "trade_date": trade_dates,
                "daily_return": [0.001] * 40,
            }
        ),
    )

    monkeypatch.setattr(service, "resolve_base_factor_codes", lambda base_factors: ["close"])
    monkeypatch.setattr(
        service,
        "generate_candidate_expressions",
        lambda **kwargs: ["rank(close)"],
    )
    monkeypatch.setattr(
        service,
        "_filter_supported_expressions",
        lambda expressions, *, sample_frames, limit: expressions[:limit],
    )
    monkeypatch.setattr(
        factor_service.calculator,
        "calculate",
        lambda df, expr: pd.to_numeric(df["close"], errors="coerce").astype(float),
    )
    monkeypatch.setattr(
        service,
        "_write_candidate_report",
        lambda strategy_returns, benchmark_returns, periods_per_year: (
            {"sharpe": 1.0, "max_drawdown": 0.1},
            "/api/mining/reports/mock.html",
        ),
    )

    result = service.run_auto_mining(
        prompt="提升量价复合因子的稳定性",
        base_factors=["close"],
        start_date="2024-01-01",
        end_date="2024-02-29",
        universe="hs300",
        benchmark="hs300",
        n_groups=2,
        holding_period=1,
        n_candidates=1,
    )

    assert result["best_score"] >= 0
    assert len(result["factors"]) == 1
    assert result["factors"][0]["expression"] == "rank(close)"


def test_factor_calculator_supports_integer_volume_sma() -> None:
    df = pd.DataFrame(
        {
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000 + i * 10 for i in range(40)],
            "amount": [10000 + i * 100 for i in range(40)],
        }
    )

    volume_sma = factor_service.calculator.calculate(df.copy(), "SMA(volume, timeperiod=5)")
    composite = factor_service.calculator.calculate(
        df.copy(),
        "((SMA(close, timeperiod=5) - SMA(close, timeperiod=20)) / SMA(close, timeperiod=20)) * (volume / SMA(volume, timeperiod=20))",
    )

    assert int(volume_sma.dropna().shape[0]) > 0
    assert int(composite.dropna().shape[0]) > 0


def test_quantgpt_expression_engine_supports_integer_volume_sma() -> None:
    service = AutoFactorMiningService()
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "stock_code": ["AAA"] * 40,
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000 + i * 10 for i in range(40)],
            "amount": [10000 + i * 100 for i in range(40)],
        }
    )

    volume_sma = service._quantgpt_engine.execute_on_panel(df.copy(), "SMA(volume, timeperiod=5)").factor_series
    composite = service._quantgpt_engine.execute_on_panel(
        df.copy(),
        "((SMA(close, timeperiod=5) - SMA(close, timeperiod=20)) / SMA(close, timeperiod=20)) * (volume / SMA(volume, timeperiod=20))",
    ).factor_series

    assert volume_sma is not None
    assert composite is not None
    assert int(volume_sma.dropna().shape[0]) > 0
    assert int(composite.dropna().shape[0]) > 0


def test_quantgpt_expression_engine_supports_scientific_notation_constants() -> None:
    service = AutoFactorMiningService()
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "stock_code": ["AAA"] * 40,
            "open": [10.0 + i for i in range(40)],
            "high": [10.5 + i for i in range(40)],
            "low": [9.5 + i for i in range(40)],
            "close": [10.2 + i for i in range(40)],
            "volume": [1000 + i * 10 for i in range(40)],
            "amount": [10000 + i * 100 for i in range(40)],
        }
    )

    factor_series = service._quantgpt_engine.execute_on_panel(
        df.copy(),
        "rank(ts_mean(volume, 10) / (ts_std(volume, 10) + 1e-6))",
    ).factor_series

    assert factor_series is not None
    assert len(factor_series) == len(df)
    assert int(factor_series.dropna().shape[0]) > 0


def test_quantgpt_expression_engine_keeps_ts_corr_aligned_with_panel_rows() -> None:
    service = AutoFactorMiningService()
    rows = []
    for stock_code, offset in (("AAA", 0.0), ("BBB", 5.0)):
        for day in range(20):
            rows.append(
                {
                    "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "stock_code": stock_code,
                    "open": 10.0 + offset + day,
                    "high": 10.5 + offset + day,
                    "low": 9.5 + offset + day,
                    "close": 10.2 + offset + day,
                    "volume": 1000.0 + offset * 10 + day * 20,
                    "amount": 10000.0 + offset * 100 + day * 200,
                }
            )
    panel_df = pd.DataFrame(rows)

    factor_series = service._quantgpt_engine.execute_on_panel(
        panel_df.copy(),
        "ts_corr(close, volume, 5)",
    ).factor_series

    assert factor_series is not None
    assert len(factor_series) == len(panel_df)
    assert list(factor_series.index) == list(panel_df.index)
