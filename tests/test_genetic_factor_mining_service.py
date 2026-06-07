from __future__ import annotations

import random

import pandas as pd

from backend.services.genetic_factor_mining_service import GeneticFactorMiningService


class _StubCalculator:
    def calculate(self, data: pd.DataFrame, expression: str) -> pd.Series:
        return data["close"]


def test_single_base_factor_generates_diverse_expressions() -> None:
    random.seed(7)
    data = pd.DataFrame(
        {
            "close": [10.0, 10.5, 11.0, 10.8, 11.2, 11.6, 11.1, 11.4],
            "open": [9.9, 10.2, 10.7, 10.7, 11.0, 11.3, 10.9, 11.2],
            "high": [10.2, 10.7, 11.2, 10.9, 11.4, 11.8, 11.3, 11.6],
            "low": [9.8, 10.1, 10.8, 10.5, 10.9, 11.2, 10.8, 11.1],
            "volume": [100, 110, 120, 130, 125, 140, 135, 145],
            "return": [0.0, 0.05, 0.04, -0.02, 0.03, 0.02, -0.04, 0.03],
        }
    )

    service = GeneticFactorMiningService(
        base_factors=["close"],
        data=data,
        return_column="return",
        population_size=6,
        n_generations=2,
        cx_prob=0.7,
        mut_prob=0.3,
        factor_calculator=_StubCalculator(),
    )

    expressions = {service._generate_random_individual()[0] for _ in range(20)}

    assert "factor_0" in expressions
    assert len(expressions) >= 4
    assert any("shift" in expression or "rolling" in expression or "diff" in expression for expression in expressions)
