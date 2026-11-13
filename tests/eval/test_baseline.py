from sales_agents.agents.baseline import make_rule_baseline
from sales_agents.datagen import generate_leads, template_text
from sales_agents.eval.harness import run_eval


def test_rule_baseline_is_perfect_because_labels_use_the_same_rubric():
    cases = generate_leads(60, seed=42, write_text=template_text)

    report = run_eval(cases, make_rule_baseline(cases), token_cap=100)

    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.errors == []
