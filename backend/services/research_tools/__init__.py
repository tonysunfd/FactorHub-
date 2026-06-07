"""统一研究工具层。"""

__all__ = [
    "QuantGPTClient",
    "validation_service",
    "scoring_service",
    "diagnosis_service",
    "robustness_service",
    "wqbrain_service",
    "factor_selection_service",
]


def __getattr__(name: str):
    if name == "QuantGPTClient":
        from .quantgpt_client import QuantGPTClient

        return QuantGPTClient
    if name == "validation_service":
        from .validation_service import validation_service

        return validation_service
    if name == "scoring_service":
        from .scoring_service import scoring_service

        return scoring_service
    if name == "diagnosis_service":
        from .diagnosis_service import diagnosis_service

        return diagnosis_service
    if name == "robustness_service":
        from .robustness_service import robustness_service

        return robustness_service
    if name == "wqbrain_service":
        from .wqbrain_service import wqbrain_service

        return wqbrain_service
    if name == "factor_selection_service":
        from .factor_selection_service import factor_selection_service

        return factor_selection_service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
