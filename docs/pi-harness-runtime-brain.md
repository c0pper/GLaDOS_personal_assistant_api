# Pi as the GLaDOS runtime brain

**Status:** evaluated, not implemented · **Date:** 2026-11-23 · **Scope:** runtime only (the dev-workflow use of Pi is out of scope)

Evaluation of [Pi](https://pi.dev/) (Earendil's minimal agent harness) as a replacement for the
orchestrator → tool agent → responder pipeline that answers `/api/start`.

## What Pi is

Pi is **not** a chatbot framework or a hosted service. It is a minimal, TypeScript/Node **coding-agent
harness** that runs locally:

- Agent loop with tool calling, built-in MCP, tree-structured sessions, compaction, skills, prompt
  templates, extensions and npm/git packages.
- Built-in tools are `read`, `bash`, `edit`, `write` (plus `grep`, `find`, `ls`); it has **no permission
  popups** by design.
- Four interfaces: interactive TUI, `--print`/`--mode json`, **`--mode rpc`** (JSONL over stdin/stdout),
  and a Node SDK.
- Provider coverage includes OpenRouter, so the existing `OPENROUTER_API_KEY` carries over.

For a Python host the only realistic integration is **RPC mode**. There is no maintained Python SDK; a
client is ~200 lines (Pi's docs ship a Python snippet).

## Why consider it

The current pipeline has one structural limitation: `OrchestratorOutputSchema.tool_name` is a single
`Literal`, and `run_pipeline()` invokes exactly one of `MCP_AGENTS`, Vikunja, or "No Tool". A request such as

> "turn off the kitchen light and add milk to the shopping list"

cannot be served — it needs Home Assistant *and* Grocy. Pi's agent loop chains tools across servers
naturally, and it already speaks MCP.

The rest of Pi's feature set (sessions, compaction, extensions) is mostly a different way of doing
things this repo already does, not a missing capability.

## Where it would plug in

The seam is `run_pipeline()` in `src/api/endpoints/start.py`. Everything else stays as it is:

| Component | Change |
|---|---|
| `telegram_bot/bot.py` | **None.** It still POSTs to `/api/start` and `/api/start_audio`. |
| `src/api/endpoints/start.py` | Replace the brain inside `run_pipeline()`; keep routing, auth and error handling. |
| `/start_audio`, STT/TTS (`services/speech_to_text.py`, `text_to_speech.py`) | **None.** Pi only produces the text reply. |
| `src/api/endpoints/conversations.py` | Map "reset" onto a new/rotated Pi session. |

What Pi would replace:

| Today | With Pi |
|---|---|
| `agents/orchestrator_agent.py` picks one tool and rewrites follow-ups | One agent decides, rewrites and chains tools itself |
| `agents/mcp_agent.py` hand-rolled loop, `allowed_tools` allow-lists, 8000-char truncation, round caps | Pi's built-in loop + MCP exposure controls |
| `agents/glados_responder_agent.py` re-voices the result | Tone handled by `--system-prompt` on the same model |
| `services/conversation_store.py` SQLite history | Pi session files, trees, compaction, `/export` to HTML |
| Per-agent `llm_cache` namespaces, optional Jev routing | No equivalent without writing Pi extensions |

## Gains

- **Multi-tool and multi-step requests**, the one capability genuinely missing today.
- **Native MCP** for HA/Grocy/Tavily — the servers port over directly. Vikunja is REST today
  (`agents/vikunja_agent.py`) and would need an MCP wrapper or a small extension.
- **Codemode**: a QuickJS sandbox where MCP calls can run with `Promise.allSettled`, instead of the
  sequential loop in `mcp_agent.py`.
- **Sessions on disk**: branchable conversation trees, HTML export, survives restarts.
- **One agent loop** instead of a classifier + tool loop + responder chain.

## Costs and risks

1. **Node in the runtime.** The API image is `uv:python3.14-trixie-slim`. Either add Node + `pi` to it or
   run a sidecar and bridge stdio. No maintained Python SDK, so the RPC client is ours to write and test.
2. **Process model.** RPC handles one active run per process; prompts queue. Spawning per message is too
   slow (startup + MCP handshakes), so this needs a small pool keyed by conversation with idle eviction.
   That is infrastructure the current 2-call pipeline does not have.
3. **Cost and latency become less predictable.** The cheap path today is: flash-lite classifier, one tool
   round, gpt-4o-mini responder. A coding-agent loop will spend more tokens and more turns on
   "turn off the light". Jev fast routing and the LLM cache have no drop-in equivalent.
4. **Tone becomes prompt maintenance.** Today the responder is a small, tightly constrained schema. With
   Pi you override a coding harness's system prompt (`--system-prompt` or `SYSTEM.md`) and depend on the
   model behaving like a butler.
5. **Security surface.** Pi defaults to `read`/`bash`/`edit`/`write` and asks for no confirmation. It must
   run with built-in tools disabled (or in a locked-down container). Even then, Tavily search results
   flowing into an agent that can actuate Home Assistant is a prompt-injection path: injected text could
   trigger real actions, not just a bad answer.
6. **Session IDs.** Pi accepts only letters, digits, `.`, `_`, `-`. `telegram:{chat_id}` must be
   sanitized, e.g. `telegram-123`.
7. **Shutdown and failure mapping.** Pi's lifecycle signals (`agent_settled`, retry/compaction events,
   orderly stdin-EOF shutdown) must be mapped onto the existing timeout and `UNREACHABLE_MESSAGE` paths.

## Security baseline

Run Pi with the smallest possible surface:

- `--no-builtin-tools` (no bash/edit/write) or an explicit `--tools 'mcp__*'` allow-list.
- `--no-context-files` so the container's `AGENTS.md` does not leak into the assistant's prompt.
- `--no-skills --no-prompt-templates` unless deliberately bundled.
- A dedicated empty working directory, read-only where possible.
- MCP servers limited to household read/write actions; no admin/destructive tools.
- Treat every user message and every web-search result as untrusted input to an actuating agent.

## Spike design

Behind a flag, next to the existing pipeline, so the current brain remains the default:

1. **Endpoint:** `POST /api/start_pi` (or `run_pipeline(..., brain="pi")`), same request/response shape as
   `/start` so the Telegram bot can switch with one URL.
2. **Process:** `pi --mode rpc --no-session --no-context-files --no-builtin-tools --no-skills
   --no-prompt-templates --system-prompt <glados prompt>` with the MCP servers in the config.
   (`pi --help` is authoritative for the installed version.)
3. **Sessions:** when memory is wanted, drop `--no-session` and use a sanitized id:

   ```python
   def pi_session_id(conversation_id: str) -> str:
       return "".join(c if c.isalnum() or c in "._-" else "-" for c in conversation_id)
   ```

4. **Client:** strict JSONL framing on stdin/stdout. Read stdout as bytes and split on LF only — do not
   use a line reader that also splits on U+2028/U+2029, which are legal inside JSON strings. Stderr is
   diagnostics, never protocol. The PowerShell/Node `readline` trap is documented by Pi for the same reason.
5. **Completion:** wait for `agent_settled`, not `agent_end` (retries, compaction and follow-ups can still
   follow). Subscribe to events before sending the prompt.
6. **Timeouts:** reuse `API_TIMEOUT_SECONDS` semantics; abort the run and fall back to the current brain on
   timeout, so a Pi regression cannot take the bot down.
7. **Cache:** either skip caching for the spike or implement a Pi extension that mirrors the
   `llm_cache` namespaces.

### Benchmark

Reuse the pattern in `scripts/jev_benchmark.py`. Fixed utterance set, at minimum:

- single-tool: lights, temperature, shopping list, chores, to-dos, web search;
- multi-tool: "turn off the kitchen light and add milk to the shopping list";
- follow-ups with history: "turn it off", "and tomorrow?";
- adversarial: web-search text containing injected instructions.

Measure latency (p50/p95), token cost per request, tool-selection accuracy, and whether the reply keeps
GLaDOS's tone. Compare against the current pipeline on the same machine and model budget.

## Recommendation

Do **not** replace the runtime brain wholesale. The only missing capability is multi-tool requests, so the
cheapest path is:

1. Spike `/api/start_pi` behind a flag (~1–2 days).
2. Benchmark it against the current pipeline.
3. If it wins, use it as the **multi-tool fallback** first (low orchestrator confidence or multiple
   detected intents), keeping the cheap deterministic path for the common single-tool case.
4. Only promote it to the default if it also wins on latency and cost, or if multi-tool requests become
   the normal case.

A production version with pooling, timeouts, session reset, tests and cache hooks is roughly 1–2 weeks.

## References

- Pi site: <https://pi.dev/>
- RPC mode: <https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/rpc.md>
- CLI reference: <https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/cli.md>
- MCP: <https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/mcp.md>
