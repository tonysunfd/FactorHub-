import pandas as pd

from backend.data.enrichment import MarketDataEnrichmentService, market_data_enrichment_service
from backend.data.registry import build_default_data_source_registry
from backend.data.service import DataService
from backend.services.factor_neutralization_service import FactorNeutralizationService
from backend.services.factor_service import FactorCalculator, FactorService


def test_detect_variables_covers_fundamental_and_special_fields():
    expressions = [
        "roe / pb",
        "dividend_yield + market_cap",
        "industry_code == industry_code",
    ]

    detected = market_data_enrichment_service.detect_variables(expressions)

    assert {"roe", "pb", "dividend_yield", "market_cap", "industry_code"} <= detected


def test_factor_calculator_exposes_dynamic_dataframe_columns():
    df = pd.DataFrame(
        {
            "open": [10.0, 10.5, 11.0],
            "high": [10.2, 10.8, 11.1],
            "low": [9.8, 10.2, 10.7],
            "close": [10.1, 10.6, 11.2],
            "volume": [1000, 1100, 1200],
            "roe": [0.1, 0.12, 0.14],
            "net_profit": [100.0, 110.0, 120.0],
            "total_share": [10.0, 10.0, 10.0],
        }
    )

    result = FactorCalculator().calculate(df, "close * roe")

    assert result.tolist() == [1.01, 1.272, 1.568]


def test_validate_factor_code_supports_fundamental_columns():
    is_valid, message = FactorService().validate_factor_code("roe / equity_multiplier")

    assert is_valid is True
    assert message == "验证通过"


def test_enrich_daily_data_merges_mocked_baostock_fields(monkeypatch):
    market_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "close": [10.0, 11.0],
            "volume": [1000, 1200],
        }
    )

    monkeypatch.setattr(
        market_data_enrichment_service,
        "_fetch_fundamental_daily",
        lambda stock_code, start_date, end_date, market_df, needed_vars: pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "stock_code": [stock_code, stock_code],
                "roe": [0.11, 0.12],
                "market_cap": [100.0, 120.0],
            }
        ),
    )
    monkeypatch.setattr(
        market_data_enrichment_service,
        "_fetch_dividend_yield",
        lambda stock_code, start_date, end_date, market_df: pd.Series([0.01, 0.02], index=market_df.index),
    )
    monkeypatch.setattr(
        market_data_enrichment_service,
        "get_industry_data",
        lambda stock_codes: pd.DataFrame(
            {
                "stock_code": ["sh.600000"],
                "industry": ["银行"],
                "industry_code": ["801780"],
            }
        ),
    )

    enriched = market_data_enrichment_service.enrich_daily_data(
        market_df=market_df,
        stock_code="sh.600000",
        start_date="2024-01-01",
        end_date="2024-01-05",
        needed_vars={"roe", "market_cap", "dividend_yield", "industry", "industry_code"},
    )

    assert list(enriched.index.astype(str)) == ["2024-01-02", "2024-01-03"]
    assert enriched["roe"].tolist() == [0.11, 0.12]
    assert enriched["market_cap"].tolist() == [100.0, 120.0]
    assert enriched["dividend_yield"].tolist() == [0.01, 0.02]
    assert enriched["industry"].tolist() == ["银行", "银行"]


def test_neutralization_prefers_baostock_industry_mapping(monkeypatch):
    monkeypatch.setattr(
        market_data_enrichment_service,
        "get_industry_data",
        lambda stock_codes: pd.DataFrame(
            {
                "stock_code": ["sh.600000", "sz.000001"],
                "industry": ["银行", "地产"],
            }
        ),
    )

    service = FactorNeutralizationService()
    result = service.get_industry_classification(["600000", "000001"])

    assert result == {"600000": "银行", "000001": "地产"}


def test_data_sources_report_full_baostock_capabilities():
    info = DataService().get_supported_data_sources()
    baostock = next(source for source in info["sources"] if source["key"] == "baostock")

    assert {"stock_daily", "benchmark", "universe", "fundamentals", "dividends", "industry"} <= set(baostock["capabilities"])
    assert "csi2000" in info["universes"]
    assert info["capability_priority"]["stock_daily"] == ["akshare", "baostock"]


def test_data_source_registry_orders_by_capability():
    registry = build_default_data_source_registry()

    assert [item.key for item in registry.get_sources_for_capability("stock_daily")] == ["akshare", "baostock"]
    assert [item.key for item in registry.get_sources_for_capability("benchmark")] == ["baostock"]


def test_resolve_source_order_supports_preferred_source():
    service = DataService()

    assert service._resolve_source_order("stock_daily") == ["akshare", "baostock"]
    assert service._resolve_source_order("stock_daily", preferred_source="baostock") == ["baostock", "akshare"]


def test_data_service_delegates_stock_daily_to_provider(monkeypatch):
    service = DataService()

    class StubProvider:
        def __init__(self):
            self.called = False

        def fetch_stock_daily(self, stock_code, start_date, end_date):
            self.called = True
            return pd.DataFrame(
                {
                    "日期": ["2024-01-02"],
                    "开盘": [10.0],
                    "收盘": [10.5],
                    "最高": [10.6],
                    "最低": [9.9],
                    "成交量": [1000],
                }
            )

        def to_symbol(self, stock_code):
            return stock_code

    stub = StubProvider()
    monkeypatch.setitem(service.providers, "akshare", stub)

    result = service.get_stock_data("000001.SZ", "2024-01-01", "2024-01-05", use_cache=False, preferred_source="akshare")

    assert stub.called is True
    assert list(result.columns)[:5] == ["open", "close", "high", "low", "volume"]


def test_data_service_delegates_benchmark_to_provider(monkeypatch, tmp_path):
    service = DataService()
    service.benchmark_cache_dir = tmp_path

    class StubBenchmarkProvider:
        def fetch_benchmark_rows(self, code, start_date, end_date):
            return [["2024-01-02", "100.0"], ["2024-01-03", "101.0"]]

    monkeypatch.setitem(service.providers, "baostock", StubBenchmarkProvider())

    result = service.get_benchmark_returns("hs300", "2024-01-01", "2024-01-05", preferred_source="baostock")

    assert result["benchmark"].tolist() == ["hs300", "hs300"]
    assert result["close"].tolist() == [100.0, 101.0]


def test_enrichment_service_delegates_industry_to_provider(tmp_path):
    class StubProvider:
        def fetch_industry_data(self, stock_codes):
            return pd.DataFrame(
                {
                    "stock_code": ["sh.600000"],
                    "industry": ["银行"],
                    "industry_code": ["801780"],
                }
            )

    service = MarketDataEnrichmentService(provider=StubProvider())
    service.industry_cache_dir = tmp_path
    result = service.get_industry_data(["sh.600000"])

    assert result is not None
    assert result.iloc[0]["industry"] == "银行"


def test_enrichment_service_delegates_fundamentals_to_provider(tmp_path):
    class StubProvider:
        def fetch_fundamental_quarters(self, stock_code, start_date, end_date, apis, api_func_map, api_fields):
            return pd.DataFrame(
                {
                    "code": ["sh.600000"],
                    "pubDate": ["2024-01-01"],
                    "statDate": ["2023-12-31"],
                    "roeAvg": ["0.12"],
                    "netProfit": ["100.0"],
                    "totalShare": ["10.0"],
                }
            )

    service = MarketDataEnrichmentService(provider=StubProvider())
    service.fundamental_cache_dir = tmp_path
    result = service._fetch_fundamental_quarters("sh.600000", "2024-01-01", "2024-01-05", {"profit"})

    assert result is not None
    assert result.iloc[0]["roe"] == 0.12


def test_enrichment_service_delegates_dividends_to_provider():
    class StubProvider:
        def fetch_dividend_events(self, stock_code, start_date, end_date):
            return pd.DataFrame(
                {
                    "stock_code": [stock_code],
                    "ex_date": [pd.Timestamp("2024-01-01")],
                    "cash_per_share": [0.5],
                }
            )

    service = MarketDataEnrichmentService(provider=StubProvider())
    result = service._fetch_dividend_events("sh.600000", "2024-01-01", "2024-01-05")

    assert result is not None
    assert result.iloc[0]["cash_per_share"] == 0.5
