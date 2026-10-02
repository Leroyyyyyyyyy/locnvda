"""CPU-only demonstration server, NOT an inference engine or a benchmark target.

Run locally: python -m uvicorn gateway.mock_upstream:app --port 8000
"""
import asyncio
import json

from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse

app = FastAPI(title="Local mock vLLM (development only)")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "mock-model", "object": "model", "owned_by": "mock"}]}


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def complete(request: Request):
    payload = await request.json()
    chat = request.url.path == "/v1/chat/completions"
    pieces = ["你好", "，这是", "本地模拟响应。"]
    common = {"id": "mock-response", "model": "mock-model", "created": 0}
    usage = {"prompt_tokens": 0, "completion_tokens": 3, "total_tokens": 3}
    if not payload.get("stream"):
        choice = {"index": 0, "finish_reason": "stop"}
        if chat:
            choice["message"] = {"role": "assistant", "content": "".join(pieces)}
        else:
            choice["text"] = "".join(pieces)
        return {**common, "object": "chat.completion" if chat else "text_completion",
                "choices": [choice], "usage": usage}

    async def events():
        def encode(data):
            return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

        for piece in pieces:
            choice = {"index": 0, "finish_reason": None}
            choice["delta" if chat else "text"] = {"content": piece} if chat else piece
            yield encode({**common, "object": "chat.completion.chunk" if chat else "text_completion",
                          "choices": [choice]})
            await asyncio.sleep(0.3)
        choice = {"index": 0, "finish_reason": "stop", "delta" if chat else "text": {} if chat else ""}
        yield encode({**common, "choices": [choice]})
        if payload.get("stream_options", {}).get("include_usage"):
            yield encode({**common, "choices": [], "usage": usage})
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"cache-control": "no-cache"})
