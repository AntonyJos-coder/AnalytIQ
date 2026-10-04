import pandas as pd
import pytest

from backend.analytics.csv_analyzer import load_csv
from backend.analytics.engine import AnalyticsEngine
from backend.pipeline import build_retrieval_query
from backend.router import QuestionRouter

ROWS = [
    ("2024-07-15", "A", "North", 600), ("2024-07-15", "A", "South", 400),
    ("2024-07-15", "B", "North", 600), ("2024-07-15", "B", "South", 400),
    ("2024-07-15", "C", "North", 300), ("2024-07-15", "C", "South", 200),
    ("2024-08-15", "A", "North", 700), ("2024-08-15", "A", "South", 400),
    ("2024-08-15", "B", "North", 380), ("2024-08-15", "B", "South", 350),
    ("2024-08-15", "C", "North", 300), ("2024-08-15", "C", "South", 150),
]


@pytest.fixture()
def analyzer(tmp_path):
    df = pd.DataFrame(ROWS, columns=["Date", "Product", "Region", "Revenue"])
    df["Product"] = "Product " + df["Product"]
    path = tmp_path / "sales.csv"
    df.to_csv(path, index=False)
    return load_csv(path, "sales.csv")


def test_total_for_month_is_exact(analyzer):
    res = analyzer.analyze("What was total revenue in August?")
    assert res["intent"] == "scalar" and "2,280.00" in res["answer"]
    assert res["data"][0]["value"] == 2280.0


def test_largest_decline_product(analyzer):
    res = analyzer.analyze("Which product had the largest revenue decline?")
    assert "Product B" in res["answer"] and "-27.0%" in res["answer"]
    assert res["focus"]["entities"] == ["Product B"] and res["focus"]["direction"] == "decline"
    assert res["chart"] is not None


def test_why_question_finds_biggest_contributor(analyzer):
    res = analyzer.analyze("Why did sales fall in August?")
    assert res["focus"]["entities"] == ["Product B"]
    assert "-8.8%" in res["answer"]  # 2500 -> 2280


def test_filter_by_named_entity(analyzer):
    res = analyzer.analyze("Why did Product B revenue fall in August?")
    assert res["focus"]["entities"] == ["Product B"] and "-27.0%" in res["answer"]


def test_ranking_and_trend(analyzer):
    top = analyzer.analyze("Which product had the highest revenue?")
    assert top["focus"]["entities"] == ["Product A"]
    trend = analyzer.analyze("Show monthly revenue trend")
    assert trend["intent"] == "trend" and trend["chart"]["data"][0]["type"] == "scatter"
    assert len(trend["data"]) == 2


def test_multi_hop_region_then_best_product(analyzer):
    res = analyzer.analyze("Which region had the weakest revenue growth and what was its best-selling product?")
    assert res["intent"] == "multihop"
    assert res["focus"]["entities"] == ["South", "Product A"]
    assert len(res["steps"]) == 2


def test_retrieval_query_built_from_analytics(analyzer):
    res = analyzer.analyze("Which product had the largest revenue decline in August?")
    query = build_retrieval_query("q", res["focus"])
    assert "Product B" in query and "decline" in query and "August" in query


def test_engine_picks_table_and_reports_missing_data(analyzer):
    engine = AnalyticsEngine(llm=None)
    assert engine.analyze("total revenue")["ok"] is False
    engine.add("d1", [analyzer])
    assert engine.analyze("total revenue")["ok"] is True


@pytest.mark.parametrize("question,expected", [
    ("What does the August report say about inventory?", "RAG"),
    ("What was total August revenue?", "ANALYTICS"),
    ("Which product declined most and why did management say it happened?", "HYBRID"),
    ("Why did sales fall in August?", "HYBRID"),
    ("Show monthly revenue trend", "ANALYTICS"),
])
def test_router_rules(question, expected):
    assert QuestionRouter(llm=None).route(question).route == expected


def test_router_respects_availability_and_followups():
    router = QuestionRouter(llm=None)
    assert router.route("Why did sales fall?", has_docs=False, has_tables=True).route == "ANALYTICS"
    assert router.route("Total revenue?", has_docs=True, has_tables=False).route == "RAG"
    assert router.route("Why?", previous_route="HYBRID").route == "HYBRID"
