"""The Tilth agent: Gemini + function calling + RAG tool."""
from __future__ import annotations

import time

from google import genai
from google.genai import types

import rag
from tools import TOOLS

# Tried in order; the next one is used if the current model is rejected (404/400).
MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]
MODEL = MODELS[0]

SYSTEM_PROMPT = """You are Tilth, a practical crop advisor for farmers in Punjab, Pakistan.

How to work:
1. Understand what the farmer needs and which crop, field size and city are involved.
   A "Farmer profile" line may appear at the start of a message; use it as known context.
2. If a detail that changes the answer is missing (for example crop or acres), ask one short question.
3. Pick the right tool instead of guessing:
   - search_crop_guides for agronomy facts, diseases, pests, irrigation, soil.
   - fertilizer_plan for any fertilizer quantity.
   - sowing_window for timing questions.
   - get_weather_forecast before advising on spraying, irrigation or harvesting.
   You may call several tools in one answer.
4. Ground every claim in tool results. Never invent doses, prices or dates.
5. Reply in the same language the farmer used (English, Urdu or Roman Urdu).
   Keep it short, with simple numbered steps and units farmers use (acre, 50 kg bag, maund).

Safety rules:
- For pesticides, never give a dose from memory. Tell the farmer to follow the product label,
  wear protection, and confirm with the local agriculture extension officer.
- If the question is outside farming, say so politely and steer back.
- If a tool returns an error, explain it plainly and suggest what to do next."""


def make_client(api_key: str):
    client = genai.Client(api_key=api_key)
    rag.configure(client)
    return client


def new_chat(client, model: str = MODEL):
    return client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=TOOLS,
            temperature=0.3,
        ),
    )


def build_message(question: str, profile: dict | None = None) -> str:
    """Prefix the question with the farmer profile (city, crop, acres) if set."""
    parts = [f"{k}={v}" for k, v in (profile or {}).items() if v]
    return f"[Farmer profile: {', '.join(parts)}]\n{question}" if parts else question


def extract_trace(response) -> list[dict]:
    trace = []
    for content in getattr(response, "automatic_function_calling_history", None) or []:
        for part in content.parts or []:
            if part.function_call:
                trace.append({"step": "call", "tool": part.function_call.name,
                              "args": dict(part.function_call.args or {})})
            elif part.function_response:
                trace.append({"step": "result", "tool": part.function_response.name,
                              "output": part.function_response.response})
    return trace


def ask(chat, question: str, profile: dict | None = None) -> tuple[str, list[dict]]:
    """Send one question. Returns (answer, tool_trace). Retries once on a busy API."""
    message = build_message(question, profile)
    last_error = None
    for attempt in range(2):
        try:
            response = chat.send_message(message)
            answer = response.text or "Sorry, I could not produce an answer. Please try again."
            return answer, extract_trace(response)
        except Exception as exc:  # network, quota, overload
            last_error = exc
            text = str(exc)
            if attempt == 0 and any(c in text for c in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                time.sleep(3)
                continue
            break
    raise last_error


def describe_error(exc: Exception) -> str:
    """Short, readable error text (code, status, message) for the UI."""
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    message = getattr(exc, "message", None) or str(exc)
    head = " ".join(str(x) for x in (code, status) if x)
    text = f"{head}: {message}" if head else message
    return " ".join(text.split())[:350]


class Advisor:
    """Holds the Gemini chat and falls back to another model if one is rejected."""

    def __init__(self, api_key: str):
        self.client = make_client(api_key)
        self.model_index = 0
        self.chat = new_chat(self.client, MODELS[0])

    @property
    def model(self) -> str:
        return MODELS[self.model_index]

    def ask(self, question: str, profile: dict | None = None) -> tuple[str, list[dict]]:
        while True:
            try:
                return ask(self.chat, question, profile)
            except Exception as exc:
                code = getattr(exc, "code", None)
                key_problem = "API key" in str(exc) or code in (401, 403)
                if code in (400, 404) and not key_problem and self.model_index + 1 < len(MODELS):
                    self.model_index += 1
                    self.chat = new_chat(self.client, self.model)
                    continue
                raise
