# OpenClaw Auto-Router Setup

Semantic intent-based routing for OpenClaw requests via litellm proxy.

## Routing Rules

| Intent | Target | Model(s) |
|--------|--------|----------|
| Heartbeat / health checks | Local Ollama | `llama3.2` |
| Coding tasks | OpenRouter (load-balanced) | kimi-k2, deepseek-v4, codex-mini |
| Skill / tool usage | Anthropic | Claude Sonnet 4.6 |
| General (fallback) | Google | Gemini 2.5 Flash |

## Prerequisites

```bash
# Local embedding model (used by the router for classification)
ollama pull nomic-embed-text

# Local chat model for heartbeat responses
ollama pull llama3.2

# Python dependency
pip install semantic-router
```

## Environment Variables

```bash
export OPENROUTER_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export LITELLM_MASTER_KEY=sk-your-secret
```

## Running

```bash
litellm --config routing_configs/config.yaml
```

Client usage:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "openclaw-router", "messages": [{"role": "user", "content": "write a python function to sort a list"}]}'
```

## How It Works

1. Client sends a request with `model: "openclaw-router"`
2. The auto-router extracts the last user message
3. It embeds the message using `ollama/nomic-embed-text` (local, ~10-50ms)
4. Cosine similarity is computed against the utterances defined in `routes.json`
5. The best-matching route's `name` becomes the target model group
6. If no route exceeds its `score_threshold`, the request falls back to `general` (Gemini 2.5 Flash)
7. For model groups with multiple deployments (e.g., `coding`), litellm load-balances across them

## Files

- `config.yaml` — litellm proxy config (model list, router wiring, settings)
- `routes.json` — semantic route definitions (utterances, thresholds)

## Tuning

### Utterances (`routes.json`)

The `utterances` array is the training data for each route. Add examples that represent real OpenClaw traffic:

- More utterances = better coverage
- Diverse phrasing helps (don't just repeat the same sentence with synonyms)
- 10-20 utterances per route is a good starting point

### Score Threshold

Each route has a `score_threshold` (0.0 to 1.0):

- **Higher** (e.g., 0.8) = stricter matching, fewer false positives, more requests fall through to default
- **Lower** (e.g., 0.5) = more aggressive matching, captures more edge cases but risks misrouting

Current settings:
- `heartbeat`: 0.75 (strict — these are very distinct messages)
- `coding`: 0.55 (moderate — coding requests vary widely)
- `skill`: 0.55 (moderate)

### OpenRouter Model IDs

Verify the exact model slugs on [OpenRouter's models page](https://openrouter.ai/models). The IDs in `config.yaml` may need updating:

```yaml
# Check and update these if they 404:
openrouter/moonshotai/kimi-k2
openrouter/deepseek/deepseek-chat-v4-0324
openrouter/openai/codex-mini
```

### Switching Embedding Model

Any dense embedding model works. Options:

| Model | Dimensions | Location | Cost |
|-------|-----------|----------|------|
| `ollama/nomic-embed-text` | 768 | Local | Free |
| `ollama/all-minilm` | 384 | Local | Free |
| `ollama/mxbai-embed-large` | 1024 | Local | Free |
| `openai/text-embedding-3-small` | 1536 | API | $0.02/1M tokens |

For 4 well-separated categories, even the smallest model (384 dims) works fine.
