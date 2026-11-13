import pytest

from sales_agents.datagen import generate_leads, template_text
from sales_agents.eval.harness import TokenCapExceeded, format_table, run_eval
from sales_agents.schemas import QualificationResult

CASES = generate_leads(30, seed=5, write_text=template_text)
GOLD_BY_ID = {c.lead.lead_id: c.gold for c in CASES}


def perfect(lead):
    return QualificationResult(label=GOLD_BY_ID[lead.lead_id], reason="oracle")


def test_error_counts_as_wrong_not_dropped():
    hot_ids = [c.lead.lead_id for c in CASES if c.gold == "hot"]
    broken = hot_ids[0]

    def flaky(lead):
        if lead.lead_id == broken:
            raise TimeoutError("model timed out")
        return perfect(lead)

    report = run_eval(CASES, flaky, token_cap=100)

    assert len(report.errors) == 1
    assert report.errors[0][0] == broken
    assert report.matrix[("hot", None)] == 1
    # The failed case stays in the recall denominator.
    assert report.recall == pytest.approx((len(hot_ids) - 1) / len(hot_ids))
    assert sum(report.matrix.values()) == len(CASES)


def test_invalid_output_counts_as_error():
    report = run_eval(CASES, lambda lead: {"label": "hot"}, token_cap=100)

    assert len(report.errors) == len(CASES)
    assert report.precision == 0.0 and report.recall == 0.0


def test_run_blocked_when_estimate_exceeds_cap():
    calls = []

    def spy(lead):
        calls.append(lead)
        return perfect(lead)

    with pytest.raises(TokenCapExceeded):
        run_eval(CASES, spy, token_cap=1_000, estimated_tokens=1_001)

    assert calls == []


def test_run_allowed_when_estimate_equals_cap():
    report = run_eval(CASES, perfect, token_cap=1_000, estimated_tokens=1_000)

    assert report.precision == 1.0 and report.recall == 1.0


def test_report_records_cost_and_latency():
    report = run_eval(CASES, perfect, token_cap=100, count_tokens=lambda: 1234)

    assert report.cost == 1234
    assert report.latency_p50 >= 0.0


def test_format_table_shows_name_precision_and_matrix():
    report = run_eval(CASES, perfect, token_cap=100)

    text = format_table({"oracle": report})

    assert "oracle" in text
    assert "precision" in text.lower()
    assert "hot" in text and "warm" in text and "cold" in text
