from __future__ import annotations

import pandas as pd

from backend.services.auto_factor_mining_service import AutoFactorMiningService


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
    assert context["recommended_goal"] == "ls_sharpe"
    assert "补充波动率因子" in context["suggested_actions"]


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


def test_run_auto_campaign_returns_real_rounds(monkeypatch) -> None:
    service = AutoFactorMiningService()

    def fake_run_auto_mining(**kwargs):
        base_factors = kwargs["base_factors"]
        round_index = 1 if "ExtraFactor" not in base_factors else 2
        best_score = 60 + round_index * 10
        return {
            "factors": [
                {
                    "name": f"Auto_{round_index}",
                    "expression": f"expr_{round_index}",
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

    def fake_select_factors(**kwargs):
        return {
            "selected_factors": ["ExtraFactor"],
            "selection_rationale": "补充一个新因子",
            "per_factor_reason": {"ExtraFactor": "用于下一轮探索"},
        }

    monkeypatch.setattr(service, "run_auto_mining", fake_run_auto_mining)
    monkeypatch.setattr(service, "select_factors", fake_select_factors)

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
    assert result["retained_factors"][0]["expression"] == "expr_2"
    assert snapshots[-1]["current_round"] == 2


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
    assert calls["title"] == "Factor Top-Group Backtest"
    assert calls["output_dir"] == str(service.get_report_path("placeholder.html").parent)
    assert calls["periods_per_year"] == 50
    assert metrics["sharpe"] == 1.23
    assert report_url == "/api/mining/reports/backtest_report_20260607_181500.html"
