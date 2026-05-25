"""
Minimal OpenAI-compatible inference server using transformers.
Usage:
    python3 scripts/serve_model.py --model /path/to/model --port 8002
"""
import argparse
import json
import time
import uuid
import threading
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Any
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
model = None
tokenizer = None
model_name = ""
lock = threading.Lock()


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[Any] = None
    tool_call_id: Optional[str] = None


class Tool(BaseModel):
    type: str
    function: Any


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 2048
    extra_body: Optional[Any] = None


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": model_name, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    tools = [t.model_dump() for t in req.tools] if req.tools else None

    with lock:
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        do_sample = req.temperature is not None and req.temperature > 0
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens or 2048,
                do_sample=do_sample,
                temperature=req.temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0][input_len:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)

    # Try to parse tool calls
    tool_calls = None
    content = None
    try:
        parsed = tokenizer.parse_function_calls(gen_ids, decoded) if hasattr(tokenizer, 'parse_function_calls') else None
    except Exception:
        parsed = None

    if "<tool_call>" in decoded:
        import re
        tc_matches = re.findall(r"<tool_call>(.*?)</tool_call>", decoded, re.DOTALL)
        if tc_matches:
            tool_calls = []
            for tc_str in tc_matches:
                try:
                    tc = json.loads(tc_str.strip())
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    })
                except Exception:
                    pass
        # strip tool_call tags from content
        content = re.sub(r"<tool_call>.*?</tool_call>", "", decoded, flags=re.DOTALL).strip()
        if not content:
            content = None
    else:
        # Strip special tokens
        import re
        content = re.sub(r"<\|[^|]+\|>", "", decoded).strip()
        if not content:
            content = None

    finish_reason = "tool_calls" if tool_calls else "stop"
    prompt_tokens = input_len
    completion_tokens = len(gen_ids)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def main():
    global model, tokenizer, model_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--served-model-name", default=None)
    args = parser.parse_args()

    model_name = args.served_model_name or args.model.split("/")[-1]
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    print(f"Model ready on {next(model.parameters()).device}. Serving on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
