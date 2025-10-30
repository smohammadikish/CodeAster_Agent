# llm/local_llm.py
try:
    # ✅ Import for LangChain ≥ 0.2
    from langchain_ollama import ChatOllama
except ImportError:
    # 🔙 fallback for older LangChain versions
    from langchain.chat_models import ChatOllama

def get_local_llm(model_name: str = "gemma:3:4b", temperature: float = 0.0):
    """
    Connect to the locally running Ollama model (default: gemma:3:4b)
    """
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url="http://localhost:11434",
        num_ctx=2048,
    )