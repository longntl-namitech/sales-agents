"""The command line: run one lead through the outbound path with a person at the approval step.

    uv run python -m sales_agents outbound lead-0010

The Supervisor researches and qualifies the lead, saves the label to the mock CRM
over MCP, and for a hot lead writes a draft. The draft pauses at record_outreach,
and the terminal asks you to approve, edit or reject it. Only then does the guardrail
check the text and the CRM record it. `--scripted` answers with the eval checklist
instead of asking you. Nothing is ever sent, the CRM is a mock.
"""

import argparse
import asyncio
import json
from collections.abc import Callable
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from sales_agents.agents.supervisor import LeadDesk, build_supervisor, run_outbound, scripted_decision
from sales_agents.config import get_settings
from sales_agents.datagen import load_leads
from sales_agents.tools.crm import build_crm_tools, list_crm_records, load_crm_tools
from sales_agents.tools.kb import load_docs

DEFAULT_DATASET = Path(__file__).parents[2] / "datasets" / "leads_seed42.jsonl"


def terminal_decision(request: dict, *, ask: Callable[[str], str] = input, out: Callable = print) -> dict:
    """Show the draft and turn the reviewer's answer into a resume decision."""
    args = request["args"]
    out(f"\n--- Draft awaiting your decision ---\n{args['draft_text']}\n------------------------------------")
    while True:
        answer = ask("[a]pprove, [e]dit or [r]eject? ").strip().lower()
        if answer in ("a", "approve"):
            return {"type": "approve"}
        if answer in ("r", "reject"):
            return {"type": "reject", "message": ask("Reason: ").strip() or "rejected by the reviewer"}
        if answer in ("e", "edit"):
            text = ask("New text (use \\n for a line break): ").replace("\\n", "\n")
            edited = {"name": request["name"], "args": {**args, "draft_text": text}}
            return {"type": "edit", "edited_action": edited}
        out("Please answer a, e or r.")


def run_outbound_cli(
    lead_id: str,
    *,
    decide: Callable[[dict], dict],
    dataset: Path = DEFAULT_DATASET,
    crm_db: str | Path = "crm.sqlite3",
    raw_tools: list | None = None,
    models: dict | None = None,
    out: Callable = print,
) -> dict | None:
    """Run the lead and return its CRM record. `raw_tools` and `models` let tests stand in for MCP and the model."""
    leads = {c.lead.lead_id: c.lead for c in load_leads(dataset)}
    if lead_id not in leads:
        raise SystemExit(f"unknown lead id {lead_id!r}. For example: {', '.join(list(leads)[:5])}")
    lead = leads[lead_id]
    settings = get_settings()

    async def scenario():
        tools = raw_tools if raw_tools is not None else await load_crm_tools(crm_db)
        desk = LeadDesk()
        crm_tools = build_crm_tools(tools, load_docs(), results=desk.results.get)
        agent = build_supervisor(settings, MemorySaver(), desk, models=models, crm_tools=crm_tools)
        earlier = next((r for r in await list_crm_records(tools) if r["lead_id"] == lead_id), None)
        if earlier and earlier.get("outreach"):
            out("Note: this lead already has an outreach draft from an earlier run. It stays until a new draft is approved.")
        await run_outbound(agent, desk, lead, f"cli-{lead_id}", decide=decide)
        records = await list_crm_records(tools)
        return next((r for r in records if r["lead_id"] == lead_id), None)

    out(f"Lead {lead_id}: {lead.company_name} ({lead.source})")
    record = asyncio.run(scenario())
    out(f"CRM record: {json.dumps(record, indent=2)}")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sales_agents", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    outbound = sub.add_parser("outbound", help="run one lead through research, qualification, draft, approval and the CRM")
    outbound.add_argument("lead_id")
    outbound.add_argument("--scripted", action="store_true", help="answer approvals with the eval checklist instead of asking")
    outbound.add_argument("--crm-db", default="crm.sqlite3", help="sqlite file for the mock CRM (default crm.sqlite3)")
    args = parser.parse_args(argv)
    decide = scripted_decision(load_docs()) if args.scripted else terminal_decision
    run_outbound_cli(args.lead_id, decide=decide, crm_db=args.crm_db)


if __name__ == "__main__":
    main()
