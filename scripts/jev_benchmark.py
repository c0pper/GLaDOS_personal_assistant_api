"""
Compare Jev routing with the current orchestrator LLM on a small labelled set.

Usage:
    uv run python scripts/jev_benchmark.py
    uv run python scripts/jev_benchmark.py --messages my_messages.json

The optional JSON file is a list of {"message": ..., "history": ..., "expected": ...}.
Requires a valid OPENROUTER_API_KEY in .env.
"""
from pathlib import Path
import argparse
import asyncio
import json
import sys
import time

from dotenv import load_dotenv
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Config reads the environment at import time, so load .env before importing src.
load_dotenv(PROJECT_ROOT / ".env")

from src.agents.orchestrator_agent import (  # noqa: E402
    TOOL_DESCRIPTIONS,
    OrchestratorInputSchema,
    build_agent,
    build_jev_request,
)
from src.config import Config  # noqa: E402
from src.services import jev_client  # noqa: E402


MESSAGES = [
    {"message": "Turn off the living room lights.", "expected": "Home Assistant Tool"},
    {"message": "What's the temperature in the bedroom?", "expected": "Home Assistant Tool"},
    {"message": "Add milk to the shopping list.", "expected": "Grocy Tool"},
    {"message": "What's about to expire in the fridge?", "expected": "Grocy Tool"},
    {"message": "Remind me to call the dentist tomorrow.", "expected": "Vikunja Tool"},
    {"message": "Show me my pending tasks.", "expected": "Vikunja Tool"},
    {"message": "Who won the football match yesterday?", "expected": "SearXNG Tool"},
    {"message": "Hello, how are you?", "expected": "No Tool"},
    {"message": "Spegni le luci del soggiorno.", "expected": "Home Assistant Tool"},
    {"message": "Cosa sta per scadere in frigo?", "expected": "Grocy Tool"},
    {"message": "Aggiungi il latte alla lista della spesa.", "expected": "Grocy Tool"},
    {"message": "Mostrami le mie attività.", "expected": "Vikunja Tool"},
    {
        "message": "Turn it off",
        "history": "User: Turn on the kitchen lights\nGLaDOS: Done.",
        "expected": "Home Assistant Tool",
    },
    {
        "message": "What about tomorrow?",
        "history": "User: What's the weather today?\nGLaDOS: Sunny.",
        "expected": "SearXNG Tool",
    },
]


def parse_jev_answer(response: jev_client.JevResponse) -> dict:
    answer = response.answers.get("tool")
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return {}
    rewrite_answer = response.answers.get("needs_rewrite")
    return {
        "tool": answer.get("choice"),
        "confidence": answer.get("confidence"),
        "needs_rewrite": rewrite_answer.get("noul") if isinstance(rewrite_answer, dict) else None,
        "cost": response.usage.cost if response.usage else None,
    }


def accepted_by_router(parsed: dict, has_history: bool) -> bool:
    tool, confidence = parsed.get("tool"), parsed.get("confidence")
    if tool not in TOOL_DESCRIPTIONS or confidence is None:
        return False
    threshold = Config.JEV_NO_TOOL_MIN_CONFIDENCE if tool == "No Tool" else Config.JEV_MIN_CONFIDENCE
    if confidence < threshold:
        return False
    if has_history:
        needs_rewrite = parsed.get("needs_rewrite")
        if needs_rewrite is None or needs_rewrite > Config.JEV_FOLLOW_UP_THRESHOLD:
            return False
    return True


async def check_openrouter_key() -> bool:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {Config.OPENROUTER_API_KEY}"},
        )
    if response.status_code != 200:
        print(f"OpenRouter rejected the key: {response.status_code} {response.text.strip()}")
        print("Update OPENROUTER_API_KEY in .env, then run this script again.")
        return False
    return True


async def run_case(case: dict) -> dict:
    message = case["message"]
    history = case.get("history")
    result = {"message": message, "expected": case.get("expected"), "history": bool(history)}

    state, questions = build_jev_request(message, history)
    started = time.perf_counter()
    try:
        response = await jev_client.decide(state=state, questions=questions)
        result["jev_ms"] = (time.perf_counter() - started) * 1000
        result["jev"] = parse_jev_answer(response)
        result["jev_accepted"] = accepted_by_router(result["jev"], bool(history))
    except jev_client.JevError as e:
        result["jev_ms"] = (time.perf_counter() - started) * 1000
        result["jev_error"] = str(e)

    started = time.perf_counter()
    try:
        route = await build_agent().run_async(
            OrchestratorInputSchema(chat_message=message, conversation_history=history)
        )
        result["llm_ms"] = (time.perf_counter() - started) * 1000
        result["llm"] = route.tool_name
        result["llm_standalone_query"] = route.standalone_query
    except Exception as e:  # noqa: BLE001 - the benchmark should report, not crash
        result["llm_ms"] = (time.perf_counter() - started) * 1000
        result["llm_error"] = repr(e)

    return result


def print_results(results: list[dict]) -> None:
    print(
        f"{'#':>2}  {'expected':<21} {'jev':<21} {'conf':>5} {'use':<3} "
        f"{'llm':<21} {'jev ms':>7} {'llm ms':>7}"
    )
    print("-" * 112)
    for i, result in enumerate(results, 1):
        jev = result.get("jev", {})
        jev_tool = jev.get("tool") or "ERROR"
        confidence = f"{jev['confidence']:.2f}" if jev.get("confidence") is not None else "-"
        accepted = "yes" if result.get("jev_accepted") else "no"
        llm = result.get("llm") or "ERROR"
        print(
            f"{i:>2}  {str(result.get('expected') or '-'):<21} {jev_tool:<21} {confidence:>5} {accepted:<3} "
            f"{llm:<21} {result.get('jev_ms', 0):>7.0f} {result.get('llm_ms', 0):>7.0f}"
        )
        for key in ("jev_error", "llm_error"):
            if result.get(key):
                print(f"    {key}: {result[key]}")

    labelled = [r for r in results if r.get("expected")]
    jev_correct = sum(r.get("jev", {}).get("tool") == r["expected"] for r in labelled)
    llm_correct = sum(r.get("llm") == r["expected"] for r in labelled)
    agree = sum(
        r.get("jev", {}).get("tool") == r.get("llm")
        for r in results
        if r.get("jev", {}).get("tool") and r.get("llm")
    )
    accepted = sum(bool(r.get("jev_accepted")) for r in results)
    jev_times = [r["jev_ms"] for r in results if "jev_ms" in r]
    llm_times = [r["llm_ms"] for r in results if "llm_ms" in r]
    cost = sum((r.get("jev", {}).get("cost") or 0) for r in results)

    print("\nSummary")
    print(f"  Jev accuracy:  {jev_correct}/{len(labelled)}")
    print(f"  LLM accuracy:  {llm_correct}/{len(labelled)}")
    print(f"  Agreement:     {agree}/{len(results)}")
    print(f"  Jev accepted:  {accepted}/{len(results)} (these would skip the LLM)")
    if jev_times:
        print(f"  Avg latency:   Jev {sum(jev_times) / len(jev_times):.0f} ms, "
              f"LLM {sum(llm_times) / len(llm_times):.0f} ms")
    print(f"  Jev cost:      ${cost:.6f}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=Path, help="JSON file with a list of test cases")
    args = parser.parse_args()

    cases = json.loads(args.messages.read_text()) if args.messages else MESSAGES
    if not await check_openrouter_key():
        raise SystemExit(1)

    jev_client.reset()
    results = [await run_case(case) for case in cases]
    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
