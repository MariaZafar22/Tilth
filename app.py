"""Tilth: Streamlit front end."""
import json
import os

import streamlit as st
from dotenv import load_dotenv

import agent

load_dotenv()
st.set_page_config(page_title="Tilth", page_icon="🌾", layout="centered")
 
st.markdown(
    """
<style>
.hero {padding:1.2rem 1.4rem;border-radius:16px;background:linear-gradient(120deg,#2E7D32,#7CB342);color:white;margin-bottom:1rem}
.hero h1 {margin:0;font-size:1.9rem;color:white}
.hero p {margin:.3rem 0 0;opacity:.95}
</style>
<div class="hero"><h1>🌾 Tilth</h1>
<p>Crop advice for Punjab farmers: guides, weather, fertilizer maths and sowing dates in one chat.</p></div>
""",
    unsafe_allow_html=True,
)

def server_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


with st.sidebar:
    st.header("Setup")
    key = server_key()
    if not key:
        key = st.text_input("Gemini API key", type="password", key="typed_key")

    st.subheader("My field (optional)")
    city = st.text_input("City or town", placeholder="Faisalabad", key="p_city")
    crop = st.selectbox("Crop", ["", "wheat", "rice", "cotton", "sugarcane", "maize"], key="p_crop")
    acres = st.number_input("Acres", min_value=0.0, value=0.0, step=0.5, key="p_acres")

    st.subheader("Try asking")
    examples = [
        "Mujhe 5 acre gandum ke liye khad ka hisaab bata do",
        "Is it a good time to sow wheat? Check the weather too.",
        "Cotton leaves are curling and turning yellow. What could it be?",
        "Can I spray tomorrow?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["queued"] = ex
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("chat_key", None)
        st.session_state["messages"] = []
        st.rerun()

if not key:
    st.info("Add your Gemini API key in the sidebar (or in Streamlit secrets) to start.")
    st.stop()

if st.session_state.get("chat_key") != key:
    st.session_state["advisor"] = agent.Advisor(key)
    st.session_state["chat_key"] = key
    st.session_state["messages"] = []

profile = {"city": city.strip(), "crop": crop, "acres": acres if acres > 0 else ""}

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔍 Agent steps"):
                for t in msg["trace"]:
                    if t["step"] == "call":
                        st.markdown(f"**Called** `{t['tool']}`")
                        st.code(json.dumps(t["args"], indent=2), language="json")
                    else:
                        st.markdown(f"**Result from** `{t['tool']}`")
                        st.code(json.dumps(t["output"], indent=2, default=str)[:1500], language="json")

prompt = st.chat_input("Ask about your crop, field or weather...") or st.session_state.pop("queued", None)
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking guides and tools..."):
            try:
                answer, trace = st.session_state["advisor"].ask(prompt, profile)
            except Exception as exc:
                detail = agent.describe_error(exc)
                low = detail.lower()
                if "api key" in low or detail.startswith(("401", "403")):
                    hint = "Your API key was rejected. Check it and try again."
                elif "429" in detail or "RESOURCE_EXHAUSTED" in detail:
                    hint = "The AI service is busy (quota reached). Wait a minute and try again."
                else:
                    hint = "The AI service could not handle this request."
                answer, trace = f"{hint}\n\n`{detail}`", []
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer, "trace": trace})
    st.rerun()
