from sales_agents.schemas import LABELS, Label


def precision_recall(
    y_true: list[Label], y_pred: list[Label | None], positive: Label = "hot"
) -> tuple[float, float]:
    """A None prediction (the agent failed) is never positive, but the case stays counted."""
    true_positive = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    predicted = sum(p == positive for p in y_pred)
    actual = sum(t == positive for t in y_true)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    return precision, recall


def confusion_matrix(
    y_true: list[Label], y_pred: list[Label | None]
) -> dict[tuple[Label, Label | None], int]:
    """The 3x3 label cells plus a None column for failed predictions."""
    matrix = {(t, p): 0 for t in LABELS for p in (*LABELS, None)}
    for true, pred in zip(y_true, y_pred):
        matrix[(true, pred)] += 1
    return matrix
