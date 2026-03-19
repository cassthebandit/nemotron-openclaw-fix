# Nemotron 3 Super + OpenClaw: LiteLLM Reasoning Guardrail

A LiteLLM `CustomGuardrail` that fixes blank responses when running NVIDIA Nemotron 3 Super (or any reasoning model) through OpenClaw via LiteLLM.

## The Problem

NVIDIA NIM duplicates reasoning tokens into both `reasoning_content` and `content` fields during streaming. OpenClaw gets confused and renders blank messages or leaks reasoning into chat.

This affects every reasoning model served through NVIDIA NIM, and is documented in:
- [NemoClaw #247](https://github.com/NVIDIA/NemoClaw/issues/247)
- [OpenClaw #27806](https://github.com/openclaw/openclaw/issues/27806)
- [LiteLLM #20246](https://github.com/BerriAI/litellm/issues/20246)

## The Fix

A ~70-line Python guardrail that deduplicates the streaming response. During the thinking phase, it suppresses the duplicate `content` field. When the answer begins, it suppresses the reasoning fields.

## Files

| File | Purpose |
|------|---------|
| `reasoning_guardrail.py` | The guardrail — mount into LiteLLM Docker container |
| `litellm_config.yaml` | LiteLLM config with guardrail registered + 4 model backends |

## Setup

1. Place both files in your home directory (or wherever LiteLLM can reach them)
2. Mount into Docker via `docker-compose.yml`:
```yaml
volumes:
  - ./litellm_config.yaml:/app/config.yaml
  - ./reasoning_guardrail.py:/app/reasoning_guardrail.py
```

3. Set OpenClaw compat flags on your LiteLLM provider models:
```json
{
  "reasoning": true,
  "compat": {
    "supportsDeveloperRole": false,
    "supportsReasoningEffort": true
  }
}
```

4. Restart: `docker compose up -d --force-recreate litellm`

## Architecture
```
Discord ← OpenClaw Gateway ← LiteLLM Proxy (:4000) ← NVIDIA NIM / OpenRouter / LM Studio
                                    ↑
                          reasoning_guardrail.py
```

## Blog Post

Full debugging story: [infer.blog](https://infer.blog) (link TBD)

## License

MIT
