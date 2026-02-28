from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI(title="Local Qwen Coding Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"


class CodeRequest(BaseModel):
    prompt: str
    mode: str = "generate"


def call_llm(prompt: str):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()["response"]


def build_prompt(user_prompt: str, mode: str):

    if mode == "fix":
        system = """
You are an expert software engineer.

Fix the provided code.
Return corrected code first, then brief explanation.
"""
    elif mode == "review":
        system = """
You are a senior code reviewer.

Analyze the code for:
- Bugs
- Performance issues
- Security problems
- Improvements
"""
    else:
        system = """
You are an expert software engineer.

Generate clean, production-quality code.
"""

    return system + "\nUser request:\n" + user_prompt


@app.post("/agent")
def coding_agent(req: CodeRequest):

    final_prompt = build_prompt(req.prompt, req.mode)
    result = call_llm(final_prompt)

    return {
        "mode": req.mode,
        "result": result
    }