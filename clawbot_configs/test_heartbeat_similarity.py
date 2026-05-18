"""
Probe cosine similarity of candidate heartbeat queries against the heartbeat
route's utterances, using the same nomic-embed-text model the auto-router uses.

Usage:
    uv run python clawbot_configs/test_heartbeat_similarity.py
    # or against a different host:
    EMBED_BASE=http://host.docker.internal:11434/v1 uv run python clawbot_configs/test_heartbeat_similarity.py
"""

import os
from typing import List

import numpy as np
from openai import OpenAI

EMBED_BASE = os.environ.get("EMBED_BASE", "http://localhost:11434/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

UTTERANCE_SETS: dict = {
    # Original config — kept for reference / regression baseline
    "v0_original": [
        "Heartbeat: OpenClaw system health check.",
        "ping",
        "pong",
        "heartbeat",
        "health check",
        "status check",
        "are you alive",
        "are you there",
        "are you online",
        "system status",
        "connection test",
        "keep alive",
        "keepalive",
        "ready check",
        "liveness probe",
        "readiness probe",
        "echo",
        "noop",
        "ack",
    ],
    # Drop generic conversational + ultra-short tokens that pull in "hello",
    # "thanks", and unrelated short queries. Keep canonical health-check terms.
    "v1_tightened": [
        "Heartbeat: OpenClaw system health check.",
        "heartbeat",
        "health check",
        "healthcheck",
        "liveness probe",
        "readiness probe",
        "system health check",
    ],
    # Add OpenClaw-shaped utterances that mirror real heartbeat messages
    # (HEARTBEAT_OK token, HEARTBEAT.md reference, workspace path).
    "v2_openclaw_shaped": [
        "Heartbeat: OpenClaw system health check.",
        "OpenClaw heartbeat ping",
        "HEARTBEAT_OK",
        "Reply HEARTBEAT_OK if nothing needs attention",
        "Read HEARTBEAT.md if it exists",
        "OpenClaw workspace heartbeat instruction",
        "heartbeat",
        "health check",
        "healthcheck",
        "liveness probe",
        "readiness probe",
    ],
    # Same as v2 but also includes a "current time" style stamp utterance,
    # since real heartbeats often include a timestamp line.
    "v3_with_time_stamp": [
        "Heartbeat: OpenClaw system health check.",
        "OpenClaw heartbeat ping",
        "HEARTBEAT_OK",
        "Reply HEARTBEAT_OK if nothing needs attention",
        "Read HEARTBEAT.md if it exists",
        "OpenClaw workspace heartbeat instruction",
        "heartbeat",
        "health check",
        "healthcheck",
        "liveness probe",
        "readiness probe",
        "Current time: 2026-05-18 21:09 UTC",
    ],
    # Drop the bare HEARTBEAT_OK token — it acts as a magnet for any code-like
    # uppercase identifier (TypeError, NoneType, etc.). Always wrap it in
    # heartbeat-shaped context.
    "v4_no_bare_token": [
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
    ],
}

HEARTBEAT_LIKE_QUERIES: List[str] = [
    "HEARTBEAT_OK",
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
    "Current time: Monday, May 18th, 2026 - 9:09 PM (UTC) / 2026-05-18 21:09 UTC",
    "healthcheck",
]

NON_HEARTBEAT_QUERIES: List[str] = [
    "write a python function that sorts a list of dicts by a key",
    "debug this TypeError: 'NoneType' object is not iterable",
    "what is in this image, describe everything you see",
    "generate a short video of a cat playing piano",
    "summarize the latest commit history of this repo",
    "hello",
    "thanks",
    # Real Telegram-wrapped chat message — includes a timestamp in the
    # envelope but the actual user query is unrelated to heartbeat.
    (
        '{\n'
        '  "chat_id": "telegram:5374747599",\n'
        '  "message_id": "558",\n'
        '  "sender_id": "5374747599",\n'
        '  "sender": "Sam H",\n'
        '  "timestamp": "Mon 2026-05-18 21:46 UTC"\n'
        '}\n\n'
        'Sender (untrusted metadata):\n'
        '{\n'
        '  "label": "Sam H (5374747599)",\n'
        '  "id": "5374747599",\n'
        '  "name": "Sam H"\n'
        '}\n\n'
        '再试试？'
    ),
]


def embed(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def evaluate(
    name: str,
    utterances: List[str],
    hb_queries: List[str],
    nh_queries: List[str],
    u_emb: np.ndarray,
    hb_emb: np.ndarray,
    nh_emb: np.ndarray,
    verbose: bool,
) -> dict:
    hb_sims = hb_emb @ u_emb.T
    nh_sims = nh_emb @ u_emb.T
    hb_max = hb_sims.max(axis=1)
    nh_max = nh_sims.max(axis=1)

    print(f"\n=== {name}  ({len(utterances)} utterances) ===")
    if verbose:
        for label, queries, sims in (
            ("HB", hb_queries, hb_sims),
            ("NH", nh_queries, nh_sims),
        ):
            for i, q in enumerate(queries):
                top = np.argmax(sims[i])
                preview = q if len(q) <= 70 else q[:67] + "..."
                print(f"  [{label}] {sims[i, top]:.3f}  {preview!r}  ↦ {utterances[top]!r}")

    print(f"  heartbeat max:    min={hb_max.min():.3f}  median={np.median(hb_max):.3f}  max={hb_max.max():.3f}")
    print(f"  non-heartbeat max: min={nh_max.min():.3f}  median={np.median(nh_max):.3f}  max={nh_max.max():.3f}")
    gap = hb_max.min() - nh_max.max()
    if gap > 0:
        thr = (hb_max.min() + nh_max.max()) / 2
        print(f"  → CLEAN separation, gap={gap:.3f}, suggested threshold={thr:.2f}")
    else:
        # Find best threshold by accuracy (heartbeat ≥ thr, non-heartbeat < thr)
        candidates = np.linspace(0.30, 0.95, 66)
        best = (0.0, 0, 0)  # (thr, correct, fp+fn)
        for thr in candidates:
            tp = int((hb_max >= thr).sum())
            tn = int((nh_max < thr).sum())
            correct = tp + tn
            errors = (len(hb_max) - tp) + (len(nh_max) - tn)
            if correct > best[1]:
                best = (thr, correct, errors)
        thr = best[0]
        tp = int((hb_max >= thr).sum())
        tn = int((nh_max < thr).sum())
        fn = len(hb_max) - tp
        fp = len(nh_max) - tn
        print(f"  → no clean gap; best threshold={thr:.2f}  TP={tp}/{len(hb_max)}  TN={tn}/{len(nh_max)}  FP={fp}  FN={fn}")
    return {"name": name, "hb_max": hb_max, "nh_max": nh_max}


def main() -> None:
    print(f"Embedding endpoint: {EMBED_BASE}")
    print(f"Embedding model:    {EMBED_MODEL}")
    verbose = os.environ.get("VERBOSE", "0") == "1"
    only = os.environ.get("ONLY")
    client = OpenAI(base_url=EMBED_BASE, api_key="sk-not-needed")

    hb_emb = embed(client, HEARTBEAT_LIKE_QUERIES)
    nh_emb = embed(client, NON_HEARTBEAT_QUERIES)

    for name, utterances in UTTERANCE_SETS.items():
        if only and name != only:
            continue
        u_emb = embed(client, utterances)
        evaluate(name, utterances, HEARTBEAT_LIKE_QUERIES, NON_HEARTBEAT_QUERIES, u_emb, hb_emb, nh_emb, verbose)


if __name__ == "__main__":
    main()
