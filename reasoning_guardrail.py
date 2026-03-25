import copy, sys
from litellm.integrations.custom_guardrail import CustomGuardrail


class ReasoningGuardrail(CustomGuardrail):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("[ReasoningGuardrail] Active", flush=True)

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict, response, request_data
    ):
        saw_answer = False
        accumulated_reasoning = []
        last_chunk = None

        try:
            async for chunk in response:
                try:
                    last_chunk = chunk

                    if not hasattr(chunk, "choices") or not chunk.choices:
                        yield chunk
                        continue

                    delta = chunk.choices[0].delta
                    if delta is None:
                        yield chunk
                        continue

                    rc = getattr(delta, "reasoning_content", None)
                    reasoning = getattr(delta, "reasoning", None)
                    content = getattr(delta, "content", None)

                    if not rc and not reasoning:
                        if content:
                            saw_answer = True
                        yield chunk
                        continue

                    if saw_answer:
                        mod = copy.deepcopy(chunk)
                        d = mod.choices[0].delta
                        d.reasoning_content = None
                        if hasattr(d, "reasoning"):
                            d.reasoning = None
                        yield mod
                        continue

                    if rc and not content:
                        accumulated_reasoning.append(rc)
                        yield chunk
                        continue

                    if reasoning and content and reasoning == content:
                        accumulated_reasoning.append(reasoning)
                        mod = copy.deepcopy(chunk)
                        d = mod.choices[0].delta
                        d.content = None
                        if not rc:
                            d.reasoning_content = reasoning
                        yield mod
                        continue

                    if content and (not reasoning or reasoning != content):
                        saw_answer = True
                        mod = copy.deepcopy(chunk)
                        d = mod.choices[0].delta
                        d.reasoning_content = None
                        if hasattr(d, "reasoning"):
                            d.reasoning = None
                        yield mod
                        continue

                    yield chunk

                except Exception as e:
                    print(f"[ReasoningGuardrail] chunk error: {e}", file=sys.stderr)
                    yield chunk

        finally:
            if not saw_answer and accumulated_reasoning and last_chunk is not None:
                full_text = "".join(accumulated_reasoning)
                print(
                    f"[ReasoningGuardrail] end-of-stream promotion: "
                    f"{len(full_text)} chars of reasoning → content",
                    flush=True,
                )
                try:
                    synth = copy.deepcopy(last_chunk)
                    d = synth.choices[0].delta
                    d.content = full_text
                    d.reasoning_content = None
                    if hasattr(d, "reasoning"):
                        d.reasoning = None
                    yield synth
                except Exception as e:
                    print(f"[ReasoningGuardrail] promotion error: {e}", file=sys.stderr)

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        return response
