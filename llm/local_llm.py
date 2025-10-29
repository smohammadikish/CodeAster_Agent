# llm/local_llm.py
try:
    # ✅ Import for LangChain ≥ 0.2
    from langchain_ollama import ChatOllama
except ImportError:
    # 🔙 fallback for older LangChain versions
    from langchain.chat_models import ChatOllama


def get_local_llm(model_name: str = "llama3", temperature: float = 0.0):
    """
    Load a local Ollama model (e.g. 'llama3', 'mistral', 'phi3', etc.)
    Requires Ollama to be installed and running locally.
    """
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        # Optional parameters — can be adjusted if needed:
        num_ctx=4096,  # context window size
        base_url="http://localhost:11434",  # Ollama default port
    )
    print(f"🧠 Local Ollama model loaded: {model_name}")
    return llm
