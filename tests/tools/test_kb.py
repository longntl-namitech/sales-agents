from sales_agents.tools.kb import search_product_docs


def test_search_returns_the_relevant_passage():
    hits = search_product_docs("how much does it cost per seat")

    assert hits and "$49" in hits[0]


def test_search_finds_crm_migration_help():
    hits = search_product_docs("moving from our old CRM")

    assert any("import tool" in hit for hit in hits)


def test_k_limits_the_number_of_results():
    assert len(search_product_docs("leads reps CRM data", k=2)) == 2


def test_unrelated_query_returns_nothing_rather_than_a_guess():
    assert search_product_docs("quantum blockchain submarines") == []
