import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load local .env if available
load_dotenv()

# Hardcoded Groq details for seamless integrated connection
DEFAULT_GROQ_KEY = ""
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

def get_provider():
    """Always return Groq."""
    return "Groq"

def get_api_key():
    """Retrieve the Groq API Key from environment."""
    env_key = os.getenv("GROQ_API_KEY")
    if env_key and "your_groq_api_key" not in env_key:
        return env_key
    return DEFAULT_GROQ_KEY

def get_model_name():
    """Retrieve the selected model from session state or environment, fallback to default."""
    try:
        if st.session_state and "selected_model" in st.session_state:
            return st.session_state["selected_model"]
    except Exception:
        pass
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

def get_llm(temperature=0.3, streaming=False, callbacks=None):
    api_key = get_api_key()
    primary_model = get_model_name()

    print("CONFIG.PY EXECUTED")
    print("MODEL:", primary_model)

    primary_llm = ChatOpenAI(
        model=primary_model,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=temperature,
        streaming=streaming,
        callbacks=callbacks
    )

    fallbacks = []

    # Fallback 1: llama-3.1-8b-instant if the primary model is not that
    if primary_model != "llama-3.1-8b-instant":
        fallback_groq1 = ChatOpenAI(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks
        )
        fallbacks.append(fallback_groq1)

    # Fallback 2: qwen/qwen3-32b if the primary model is not that
    if primary_model != "qwen/qwen3-32b":
        fallback_groq2 = ChatOpenAI(
            model="qwen/qwen3-32b",
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks
        )
        fallbacks.append(fallback_groq2)

    # Fallback 3: meta-llama/llama-4-scout-17b-16e-instruct if the primary model is not that
    if primary_model != "meta-llama/llama-4-scout-17b-16e-instruct":
        fallback_groq3 = ChatOpenAI(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks
        )
        fallbacks.append(fallback_groq3)

    # Fallback 4: gemini-2.5-flash if GEMINI_API_KEY is present
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key and "your_gemini_api_key" not in gemini_api_key:
        fallback_gemini = ChatOpenAI(
            model="gemini-2.5-flash",
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            temperature=temperature,
            streaming=streaming,
            callbacks=callbacks
        )
        fallbacks.append(fallback_gemini)

    if fallbacks:
        return primary_llm.with_fallbacks(fallbacks)
    return primary_llm

def is_api_configured():
    """API is always integrated and configured."""
    return True
