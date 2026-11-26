import numpy as np
from google.genai import types

import agent
import rag


class FakeResponse:
    text = "Use 2 bags of DAP."
    automatic_function_calling_history = [
        types.Content(role="model", parts=[types.Part.from_function_call(name="fertilizer_plan", args={"crop": "wheat", "acres": 2})]),
        types.Content(role="user", parts=[types.Part.from_function_response(name="fertilizer_plan", response={"dap": {"kg": 78}})]),
    ]


class FakeChat:
    def __init__(self):
        self.sent = None

    def send_message(self, message):
        self.sent = message
        return FakeResponse()


def test_trace_extraction():
    trace = agent.extract_trace(FakeResponse())
    assert [t["step"] for t in trace] == ["call", "result"]
    assert trace[0]["tool"] == "fertilizer_plan" and trace[0]["args"]["acres"] == 2


def test_profile_prefix():
    chat = FakeChat()
    answer, trace = agent.ask(chat, "How much urea?", {"city": "Faisalabad", "crop": "wheat", "acres": 3.0})
    assert chat.sent.startswith("[Farmer profile: city=Faisalabad, crop=wheat, acres=3.0]")
    assert answer and len(trace) == 2


def test_no_profile_no_prefix():
    assert agent.build_message("Hello", {"city": "", "crop": "", "acres": ""}) == "Hello"


def test_rag_with_fake_embeddings():
    class Emb:
        def __init__(self, v): self.values = v

    class Result:
        def __init__(self, vs): self.embeddings = [Emb(v) for v in vs]

    class Models:
        def embed_content(self, model, contents):
            vecs = []
            for t in contents:
                t = t.lower()
                vecs.append([t.count("rust"), t.count("cotton"), t.count("irrigation"), 0.1])
            return Result(vecs)

    class FakeClient:
        models = Models()

    rag.configure(FakeClient())
    hits = rag.search("wheat rust", top_k=1)
    assert "rust" in hits[0]["text"].lower()
    rag.configure(None)


def test_describe_error_and_fallback(monkeypatch):
    class Boom(Exception):
        code = 404
        status = "NOT_FOUND"
        message = "models/x is not found"

    assert agent.describe_error(Boom()) == "404 NOT_FOUND: models/x is not found"

    calls = []

    def fake_ask(chat, question, profile=None):
        calls.append(chat)
        if len(calls) == 1:
            raise Boom()
        return "ok", []

    monkeypatch.setattr(agent, "ask", fake_ask)
    monkeypatch.setattr(agent, "make_client", lambda key: object())
    monkeypatch.setattr(agent, "new_chat", lambda client, model=agent.MODEL: model)
    adv = agent.Advisor("k")
    answer, _ = adv.ask("hi")
    assert answer == "ok" and adv.model == agent.MODELS[1]
