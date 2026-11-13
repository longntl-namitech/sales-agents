from fastapi.testclient import TestClient

from sales_agents.crm_mock.app import create_app
from sales_agents.crm_mock.store import CrmStore


def test_http_api_upserts_records_outreach_and_lists():
    client = TestClient(create_app(CrmStore()))

    assert client.put("/leads/lead-1", json={"label": "hot", "notes": "gates pass"}).status_code == 200
    assert client.post("/leads/lead-1/outreach", json={"draft_text": "hello"}).status_code == 200

    records = client.get("/leads").json()
    assert len(records) == 1 and records[0]["outreach"] == "hello"
    assert client.get("/leads/nope").status_code == 404
