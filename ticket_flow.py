"""One-step LangGraph flow: send each ticket's message_body to a free OpenRouter model.

Only the merchant's own text (message_body) is sent. The expected_* columns
are never read, so the model can't see the answers.

Usage:
    python ticket_flow.py                  # all tickets
    python ticket_flow.py --limit 3        # first 3 tickets
    python ticket_flow.py --ticket SUP-48197
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

TICKETS_CSV = Path(__file__).parent / "shopify_merchant_tickets_test_set.csv"
OUT_DIR = Path(__file__).parent / "results"

# "openrouter/free" is OpenRouter's router: it picks one currently-free model per request.
# Override with OPENROUTER_MODEL (e.g. a specific "<name>:free" id) in .env or the shell.
DEFAULT_MODEL = "openrouter/free"

SYSTEM_PROMPT = (
    "You are a support assistant for Shopify merchants. "
    "Read the merchant's message and reply helpfully and concisely."
)


class TicketState(TypedDict, total=False):
    ticket_id: str
    message_body: str  # input: the merchant's own text
    response: str      # output: the model's reply
    model: str         # output: which model actually answered
    error: Optional[str]


def build_llm() -> ChatOpenAI:
    load_dotenv(Path(__file__).parent / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set. Put it in .env (see .env.example).")
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        timeout=90,
        max_retries=0,  # retries are handled in the node so we can back off on free-tier 429s
    )


def make_graph(llm):
    """START -> ask_model -> END"""

    def ask_model(state: TicketState) -> TicketState:
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(state["message_body"])]
        last_err = None
        for attempt in range(4):
            try:
                reply = llm.invoke(messages)
                meta = getattr(reply, "response_metadata", {}) or {}
                return {
                    "response": reply.content,
                    "model": meta.get("model_name") or meta.get("model") or "",
                    "error": None,
                }
            except Exception as exc:  # free models are flaky: rate limits, 5xx, empty replies
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(2 ** (attempt + 1))
        return {"response": "", "model": "", "error": last_err}

    g = StateGraph(TicketState)
    g.add_node("ask_model", ask_model)
    g.add_edge(START, "ask_model")
    g.add_edge("ask_model", END)
    return g.compile()


def read_tickets(path: Path = TICKETS_CSV):
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Deliberately keep only the id and the merchant's text.
            yield {"ticket_id": row["ticket_id"], "message_body": row["message_body"]}


def run(llm, tickets, out_path: Path) -> list[dict]:
    graph = make_graph(llm)
    OUT_DIR.mkdir(exist_ok=True)
    results = []
    with out_path.open("w", encoding="utf-8") as out:
        for t in tickets:
            final = graph.invoke(t)
            row = {
                "ticket_id": final["ticket_id"],
                "model": final.get("model", ""),
                "response": final.get("response", ""),
                "error": final.get("error"),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            results.append(row)
            status = "ERROR" if row["error"] else "ok"
            print(f"{row['ticket_id']}: {status} ({row['model'] or '-'})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N tickets")
    ap.add_argument("--ticket", help="only this ticket_id")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "responses.jsonl")
    args = ap.parse_args()

    tickets = list(read_tickets())
    if args.ticket:
        tickets = [t for t in tickets if t["ticket_id"] == args.ticket]
        if not tickets:
            sys.exit(f"No ticket {args.ticket}")
    if args.limit:
        tickets = tickets[: args.limit]

    results = run(build_llm(), tickets, args.out)
    failed = sum(1 for r in results if r["error"])
    print(f"\nDone: {len(results) - failed}/{len(results)} ok -> {args.out}")


if __name__ == "__main__":
    main()
