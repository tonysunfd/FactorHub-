"""
回测报告生成服务。
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_CSS_PATCH = """
<style>
body { margin: 15px !important; }
.container { max-width: 100% !important; display: flex; flex-wrap: wrap; gap: 0; }
.container > h1, .container > h4, .container > hr { width: 100%; flex-shrink: 0; }
#left { float: none !important; width: 62% !important; min-width: 0; margin-right: 0 !important; margin-top: -1.2rem; }
#right { float: none !important; width: 36% !important; min-width: 280px; }
#left svg { width: 100% !important; height: auto !important; }
@media (max-width: 700px) {
    #left, #right { width: 100% !important; }
}
</style>
"""


def generate_report(
    ls_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    title: str = "Factor Long-Short Backtest",
    output_dir: str | None = None,
    periods_per_year: int = 252,
) -> dict:
    """生成 QuantStats HTML 报告并提取关键指标。"""
    import quantstats as qs

    report_dir = Path(output_dir) if output_dir else (_PROJECT_ROOT / "reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    returns = ls_returns.sort_index().copy()
    returns.index = pd.to_datetime(returns.index).normalize()
    returns.name = "Strategy"

    aligned_benchmark = benchmark_returns
    if aligned_benchmark is not None:
        aligned_benchmark = aligned_benchmark.copy()
        aligned_benchmark.index = pd.to_datetime(aligned_benchmark.index).normalize()
        aligned_benchmark = aligned_benchmark.sort_index()
        aligned_benchmark = aligned_benchmark.reindex(returns.index, method="ffill")
        valid = ~aligned_benchmark.isna()
        if valid.sum() < 2:
            logger.warning("Insufficient benchmark overlap, generating report without benchmark")
            aligned_benchmark = None
        else:
            returns = returns[valid]
            aligned_benchmark = aligned_benchmark[valid]

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_path = str(report_dir / f"backtest_report_{timestamp}.html")

    qs.reports.html(
        returns,
        benchmark=aligned_benchmark,
        output=report_path,
        title=title,
        rf=0.03,
        match_dates=False,
        periods_per_year=periods_per_year,
    )

    _patch_report_css(report_path)
    logger.info("Report saved: %s", report_path)

    metrics = {
        "total_return": float(qs.stats.comp(returns)),
        "cagr": float(qs.stats.cagr(returns, periods=periods_per_year)),
        "sharpe": float(qs.stats.sharpe(returns, rf=0.03, periods=periods_per_year)),
        "sortino": float(qs.stats.sortino(returns, rf=0.03, periods=periods_per_year)),
        "max_drawdown": float(qs.stats.max_drawdown(returns)),
        "volatility": float(qs.stats.volatility(returns, periods=periods_per_year)),
        "win_rate": float(qs.stats.win_rate(returns)),
        "profit_factor": float(qs.stats.profit_factor(returns)),
    }

    if aligned_benchmark is not None:
        metrics["benchmark_total_return"] = float(qs.stats.comp(aligned_benchmark))
        metrics["benchmark_cagr"] = float(qs.stats.cagr(aligned_benchmark, periods=periods_per_year))

    return {"report_path": report_path, "metrics": metrics}


def _patch_report_css(report_path: str) -> None:
    try:
        path = Path(report_path)
        html = path.read_text(encoding="utf-8")
        if "</head>" in html:
            html = html.replace("</head>", _CSS_PATCH + "</head>", 1)
            path.write_text(html, encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to patch report CSS: %s", exc)
