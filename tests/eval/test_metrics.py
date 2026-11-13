import pytest

from sales_agents.eval.metrics import confusion_matrix, precision_recall


def test_precision_recall_hand_example():
    y_true = ["hot", "hot", "hot", "warm", "cold"]
    y_pred = ["hot", "warm", None, "hot", "cold"]

    precision, recall = precision_recall(y_true, y_pred)

    # Predicted hot twice, one right. Three truly hot, one found.
    assert precision == pytest.approx(1 / 2)
    assert recall == pytest.approx(1 / 3)


def test_no_hot_predictions_gives_zero_precision_not_a_crash():
    assert precision_recall(["hot", "cold"], ["cold", "cold"]) == (0.0, 0.0)


def test_matrix_has_all_nine_cells():
    matrix = confusion_matrix(["hot", "cold"], ["hot", "cold"])
    labels = ["hot", "warm", "cold"]

    assert all((t, p) in matrix for t in labels for p in labels)
    assert matrix[("hot", "hot")] == 1
    assert matrix[("cold", "cold")] == 1
    assert matrix[("warm", "warm")] == 0


def test_matrix_counts_missing_predictions_in_an_error_column():
    matrix = confusion_matrix(["hot", "warm"], [None, "warm"])

    assert matrix[("hot", None)] == 1
    assert sum(matrix.values()) == 2
