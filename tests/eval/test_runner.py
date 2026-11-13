import pytest

from sales_agents.eval.harness import TokenCapExceeded
from sales_agents.eval.metrics import accuracy
from sales_agents.eval.runner import run_cases


def test_a_failure_is_a_wrong_prediction_not_a_dropped_case():
    def fn(x):
        if x == "boom":
            raise TimeoutError("model timed out")
        return x

    result = run_cases(["a", "boom", "b"], fn, case_id=str, token_cap=1_000)

    assert result.predictions == ["a", None, "b"]
    assert result.errors == [("boom", "TimeoutError: model timed out")]
    assert accuracy(["a", "boom", "b"], result.predictions) == 2 / 3  # still three in the denominator


def test_run_blocked_when_the_estimate_exceeds_the_cap():
    with pytest.raises(TokenCapExceeded):
        run_cases(["a"], lambda x: x, case_id=str, token_cap=100, estimated_tokens=101)


def test_latency_and_cost_are_reported():
    result = run_cases(["a", "b"], lambda x: x, case_id=str, token_cap=100, count_tokens=lambda: 7)

    assert result.cost == 7 and result.latency_p50 >= 0.0 and result.n == 2


def test_accuracy_counts_none_as_wrong():
    assert accuracy(["x", "y"], ["x", None]) == 0.5
    assert accuracy([], []) == 0.0
