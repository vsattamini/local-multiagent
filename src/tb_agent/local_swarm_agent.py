"""
Custom Terminal-Bench agent (STUB, for the optional multi-agent variant).

Drives a local OpenAI-compatible llama-cpp server (see scripts/tb_serve_model.sh).
A minimal single-role loop is implemented; a planner+executor split is sketched in
comments for the later "multi-agent on terminal" variant.

Run with:  tb run --agent-import-path src.tb_agent.local_swarm_agent:LocalQwenAgent ...

NOTE: import paths / FailureMode member names must be confirmed against the
installed terminal_bench version (see docs/terminal_bench_plan.md "Open items").
This file is intentionally dependency-light and is NOT imported by the swarm code.
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from terminal_bench.agents.base_agent import BaseAgent, AgentResult
    from terminal_bench.agents.failure_mode import FailureMode
    from terminal_bench.terminal.tmux_session import TmuxSession
    _TB = True
except Exception:  # allows import/lint without terminal-bench installed
    _TB = False
    BaseAgent = object  # type: ignore

import urllib.request, json


def _chat(base_url: str, model: str, messages: list[dict], api_key: str) -> tuple[str, int, int]:
    """Minimal OpenAI /chat/completions call (no SDK dependency)."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "temperature": 0.2,
                         "max_tokens": 256}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    msg = out["choices"][0]["message"]["content"]
    u = out.get("usage", {})
    return msg, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


class LocalQwenAgent(BaseAgent):
    """Single-role ReAct-ish loop against a local Qwen server."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._base = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        self._key = os.environ.get("OPENAI_API_KEY", "sk-noauth")
        self._model = os.environ.get("TB_LOCAL_MODEL", "qwen2.5-coder")
        self._max_steps = int(os.environ.get("TB_MAX_STEPS", "20"))

    @staticmethod
    def name() -> str:
        return "local-qwen"

    def perform_task(self, task_description: str, session: "TmuxSession",
                     logging_dir: Path | None = None) -> "AgentResult":
        in_tok = out_tok = 0
        screen = session.capture_pane()
        for _ in range(self._max_steps):
            msg, pt, ct = _chat(self._base, self._model, [
                {"role": "system", "content":
                 "You control a Linux terminal. Output exactly ONE shell command, no prose. "
                 "Output the literal token DONE when the task is complete."},
                {"role": "user", "content":
                 f"Task:\n{task_description}\n\nCurrent terminal:\n{screen}\n\nNext command:"},
            ], self._key)
            in_tok += pt; out_tok += ct
            cmd = msg.strip().splitlines()[0].strip("`").strip() if msg.strip() else ""
            if not cmd or cmd.upper().startswith("DONE"):
                break
            session.send_keys([cmd, "Enter"], block=True, max_timeout_sec=60)
            screen = session.capture_pane()
        return AgentResult(total_input_tokens=in_tok, total_output_tokens=out_tok,
                           failure_mode=FailureMode.NONE, timestamped_markers=[])

# --- Planner+Executor variant (sketch for the multi-agent run) -----------------
# class PlannerExecutorAgent(BaseAgent):
#   step 1: planner LLM turns task_description into an ordered command list;
#   step 2: executor LLM issues/repairs each command using capture_pane() feedback.
#   This is the minimal "two specialised roles" mapping of the swarm onto a
#   sequential terminal task (roles are explicit here, unlike the emergent setup).
