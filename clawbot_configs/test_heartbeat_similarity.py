"""
Probe the heartbeat route end-to-end using the SAME code path as auto_router.py:
litellm.Router → LiteLLMRouterEncoder → semantic_router.SemanticRouter.

This avoids the discrepancy between raw cosine and semantic-router's internal
mean-of-top-k scoring — what this script reports is exactly what production sees.

Requires the OMLX server to be reachable at OMLX_API_BASE and OMLX_API_KEY in env.

Usage:
    OMLX_API_KEY=... uv run --with semantic-router --with transformers \\
        python clawbot_configs/test_heartbeat_similarity.py
"""

import asyncio
import os
from typing import List, Tuple

from litellm import Router
from litellm.router_strategy.auto_router.litellm_encoder import LiteLLMRouterEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.routers.base import Route

OMLX_API_BASE = os.environ.get("OMLX_API_BASE", "http://localhost:8000/v1")
OMLX_API_KEY = os.environ.get("OMLX_API_KEY", "")
EMBED_MODEL_NAME = "omlx/nomicai-modernbert-embed-base-bf16"

# v4 utterances — same set currently in claw_configs.yaml
HEARTBEAT_UTTERANCES: List[str] = [
    "Heartbeat: OpenClaw system health check.",
    "OpenClaw heartbeat ping",
    "Reply HEARTBEAT_OK if nothing needs attention",
    "Respond with HEARTBEAT_OK",
    "Read HEARTBEAT.md if it exists",
    "OpenClaw workspace heartbeat instruction at /sandbox/.openclaw/workspace/HEARTBEAT.md",
    "heartbeat",
    "health check",
    "healthcheck",
    "liveness probe",
    "readiness probe",
    "Current time: 2026-05-18 21:09 UTC",
]

HEARTBEAT_QUERIES: List[str] = [
    "HEARTBEAT_OK",
    "healthcheck",
    "Current time: Monday, May 18th, 2026 - 9:09 PM (UTC) / 2026-05-18 21:09 UTC",
    (
        "Read HEARTBEAT.md if it exists (workspace context). Follow it strictly. "
        "Do not infer or repeat old tasks from prior chats. If nothing needs "
        "attention, reply HEARTBEAT_OK."
    ),
    (
        "When reading HEARTBEAT.md, use workspace file "
        "/sandbox/.openclaw/workspace/HEARTBEAT.md (exact case). "
        "Do not read docs/heartbeat.md."
    ),
    # Exact production message that misrouted earlier under nomic-embed-text.
    (
        "Read HEARTBEAT.md if it exists (workspace context). Follow it strictly. Do not infer or repeat old tasks from prior chats. If nothing needs attention, reply HEARTBEAT_OK.\n"
        "When reading HEARTBEAT.md, use workspace file /sandbox/.openclaw/workspace/HEARTBEAT.md (exact case). Do not read docs/heartbeat.md.\n"
        "Current time: Wednesday, May 20th, 2026 - 12:09 AM (UTC) / 2026-05-20 00:09 UTC"
    ),
]

NON_HEARTBEAT_QUERIES: List[str] = [
    "write a python function that sorts a list of dicts by a key",
    "debug this TypeError: 'NoneType' object is not iterable",
    "what is in this image, describe everything you see",
    "generate a short video of a cat playing piano",
    "summarize the latest commit history of this repo",
    "hello",
    "thanks",
    # Real Telegram-wrapped chat message — has a timestamp but Chinese question.
    (
        '{\n'
        '  "chat_id": "telegram:5374747599",\n'
        '  "message_id": "558",\n'
        '  "sender_id": "5374747599",\n'
        '  "sender": "Sam H",\n'
        '  "timestamp": "Mon 2026-05-18 21:46 UTC"\n'
        '}\n\n'
        '再试试？'
    ),
    # Chinese-only — heartbeat is always English in this deployment.
    "再试试",
    "帮我写一个 Python 脚本来分析这个 CSV 文件",
    "今天天气怎么样？",
    "总结一下今天的新闻",
    "这段代码有什么 bug？",
]


def build_routelayer(score_threshold: float) -> SemanticRouter:
    """Mirror auto_router.py:152-163 — build the SemanticRouter with a real
    litellm.Router-backed encoder pointed at OMLX."""
    router = Router(
        model_list=[
            {
                "model_name": EMBED_MODEL_NAME,
                "litellm_params": {
                    "model": "openai/nomicai-modernbert-embed-base-bf16",
                    "api_base": OMLX_API_BASE,
                    "api_key": OMLX_API_KEY,
                },
            }
        ]
    )
    routes = [
        Route(
            name="heartbeat",
            description="OpenClaw heartbeat / liveness probe pings",
            utterances=HEARTBEAT_UTTERANCES,
            score_threshold=score_threshold,
        )
    ]
    return SemanticRouter(
        routes=routes,
        encoder=LiteLLMRouterEncoder(
            litellm_router_instance=router,
            model_name=EMBED_MODEL_NAME,
        ),
        auto_sync="local",
    )


def score_query(routelayer: SemanticRouter, query: str) -> Tuple[str, float]:
    """Return (matched_route_name_or_'-', similarity_score)."""
    rc = routelayer(text=query)
    name = rc.name if rc and rc.name else "-"
    score = float(rc.similarity_score) if rc and rc.similarity_score is not None else 0.0
    return name, score


async def main() -> None:
    print(f"OMLX endpoint: {OMLX_API_BASE}")
    print(f"Embedding model: {EMBED_MODEL_NAME}")
    print()

    # Build with threshold=0 so every query returns a score (we want to see
    # the raw distribution before picking a real threshold).
    routelayer = build_routelayer(score_threshold=0.0)

    hb_scores: List[float] = []
    nh_scores: List[float] = []

    print("=== HEARTBEAT queries (should match) ===")
    for q in HEARTBEAT_QUERIES:
        _, score = score_query(routelayer, q)
        hb_scores.append(score)
        preview = q.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"  {score:.4f}  {preview}")

    print("\n=== NON-HEARTBEAT queries (should NOT match) ===")
    for q in NON_HEARTBEAT_QUERIES:
        _, score = score_query(routelayer, q)
        nh_scores.append(score)
        preview = q.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"  {score:.4f}  {preview}")

    hb_min, hb_max = min(hb_scores), max(hb_scores)
    nh_min, nh_max = min(nh_scores), max(nh_scores)
    print(f"\nheartbeat:     min={hb_min:.4f}  max={hb_max:.4f}")
    print(f"non-heartbeat: min={nh_min:.4f}  max={nh_max:.4f}")
    gap = hb_min - nh_max
    if gap > 0:
        thr = (hb_min + nh_max) / 2
        print(f"→ CLEAN separation, gap={gap:.4f}, suggested threshold={thr:.2f}")
    else:
        print(f"→ no clean gap (overlap={-gap:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
