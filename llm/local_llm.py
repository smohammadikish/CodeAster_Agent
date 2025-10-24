# llm/local_llm.py
from langchain.chat_models import ChatOllama


def get_local_llm(model_name: str = "llama3"):
    """Use a locally running Ollama model for offline testing."""
    return ChatOllama(model=model_name, temperature=0)
