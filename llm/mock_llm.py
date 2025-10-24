# llm/mock_llm.py
from langchain_core.language_models import BaseLanguageModel


class MockLLM(BaseLanguageModel):
    """A fake offline LLM for local pipeline testing (no API calls)."""

    def _call(self, messages, stop=None):
        # Return a simple predictable message
        return "🤖 MockLLM: pipeline logic works fine ✅"

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "mock"
