from typing import Any, List, Optional, Sequence, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult


class MockLLM(BaseLanguageModel):
    """Offline mock LLM for testing LangChain agents without API calls."""

    # Explicit fields so Pydantic won't block assignment
    _responses: List[str] = []
    _index: int = 0

    def __init__(self, responses: Optional[List[str]] = None, **kwargs: Any):
        super().__init__(**kwargs)
        object.__setattr__(
            self,
            "_responses",
            responses
            or [
                "OK. Let's start the analysis.",
                "Geometry created.",
                "Mesh generated.",
                "Simulation completed.",
                "Results visualized.",
            ],
        )
        object.__setattr__(self, "_index", 0)

    def _next_text(self) -> str:
        """Return the next canned response."""
        text = self._responses[self._index % len(self._responses)]
        object.__setattr__(self, "_index", self._index + 1)
        return text

    # ----------------------------------------------------------
    # 🧠 Basic prediction methods
    # ----------------------------------------------------------
    def predict(self, text: str, **kwargs: Any) -> str:
        """Return the next mock response for text prediction."""
        return self._next_text()

    async def apredict(self, text: str, **kwargs: Any) -> str:
        """Async version of predict."""
        return self.predict(text, **kwargs)

    # ----------------------------------------------------------
    # 🧠 Generation methods for LangChain
    # ----------------------------------------------------------
    def generate(self, prompts: List[str], **kwargs: Any) -> LLMResult:
        """Simulate a full LLMResult generation cycle with ChatGeneration output."""
        text = self._next_text()
        return LLMResult(
            generations=[
                [ChatGeneration(message=AIMessage(content=text))] for _ in prompts
            ]
        )

    def generate_prompt(self, prompts: List[Any], **kwargs: Any) -> LLMResult:
        """Handle prompt objects as input."""
        return self.generate([str(p) for p in prompts], **kwargs)

    async def agenerate(self, prompts: List[str], **kwargs: Any) -> LLMResult:
        """Async version of generate."""
        return self.generate(prompts, **kwargs)

    async def agenerate_prompt(self, prompts: List[Any], **kwargs: Any) -> LLMResult:
        """Async version of generate_prompt."""
        return self.generate_prompt(prompts, **kwargs)

    # ----------------------------------------------------------
    # 🧠 Message-based predictions (for chat-style interaction)
    # ----------------------------------------------------------
    def predict_messages(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> BaseMessage:
        """Return the next mock message response."""
        return AIMessage(content=self._next_text())

    async def apredict_messages(
        self, messages: Sequence[BaseMessage], **kwargs: Any
    ) -> BaseMessage:
        """Async version of predict_messages."""
        return self.predict_messages(messages, **kwargs)

    # ----------------------------------------------------------
    # 🧠 Invoke methods for LangChain executors
    # ----------------------------------------------------------
    def invoke(
        self, input: Union[str, dict, List[BaseMessage]], *args: Any, **kwargs: Any
    ) -> str:
        """Return plain text (LangChain expects text-compatible output)."""
        return self._next_text()

    async def ainvoke(
        self, input: Union[str, dict, List[BaseMessage]], *args: Any, **kwargs: Any
    ) -> str:
        """Async version of invoke."""
        return self._next_text()

    # ----------------------------------------------------------
    # 🧠 Required for LangChain tool binding
    # ----------------------------------------------------------
    def bind_tools(self, tools: list, **kwargs: Any):
        """Mock method for tool binding (LangChain expects this in LLMs)."""
        return self

    # ----------------------------------------------------------
    # 🧩 Metadata
    # ----------------------------------------------------------
    @property
    def _llm_type(self) -> str:
        """Identify this mock LLM type."""
        return "mock"
