import os

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

# ✅ بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()


class GoogleGeminiLLM(BaseLanguageModel):
    """Custom LangChain-compatible wrapper for Google Gemini API (for LangChain 0.2.x)."""

    model_name: str = "gemini-1.5-flash"
    api_key: str | None = None
    model: object | None = None

    def __init__(self, model_name="models/gemini-1.5-flash", api_key=None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        # دقت کن کلید باید دقیقاً GOOGLE_API_KEY باشه
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GOOGLE_API_KEY not set in .env file")

        # پیکربندی کتابخانه رسمی گوگل
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"✅ Google Gemini initialized with model: {self.model_name}")

    # ----------------------------------------------------------
    # 🧠 متد اصلی تولید متن
    # ----------------------------------------------------------
    def generate(self, prompts, **kwargs):
        generations = []
        for prompt in prompts:
            response = self.model.generate_content(str(prompt))
            text = getattr(response, "text", str(response))
            msg = AIMessage(content=text)
            generations.append([ChatGeneration(message=msg)])
        return LLMResult(generations=generations)

    def generate_prompt(self, prompts, **kwargs):
        return self.generate([str(p) for p in prompts], **kwargs)

    # ----------------------------------------------------------
    # 🗣 متدهای پیش‌بینی (برای LangChain)
    # ----------------------------------------------------------
    def predict(self, text, **kwargs):
        response = self.model.generate_content(text)
        return getattr(response, "text", str(response))

    async def apredict(self, text, **kwargs):
        return self.predict(text)

    def predict_messages(self, messages, **kwargs) -> BaseMessage:
        text = "\n".join(m.content for m in messages if hasattr(m, "content"))
        response = self.model.generate_content(text)
        return AIMessage(content=getattr(response, "text", str(response)))

    async def apredict_messages(self, messages, **kwargs) -> BaseMessage:
        return self.predict_messages(messages, **kwargs)

    def agenerate_prompt(self, prompts, **kwargs):
        return self.generate_prompt(prompts, **kwargs)

    # ----------------------------------------------------------
    # 🔧 پشتیبانی از ابزارهای Agent
    # ----------------------------------------------------------
    def invoke(self, input, *args, **kwargs):
        if isinstance(input, dict) and "input" in input:
            input = input["input"]
        response = self.model.generate_content(str(input))
        text = getattr(response, "text", str(response))
        msg = AIMessage(content=text)
        return ChatGeneration(message=msg)

    def bind_tools(self, tools, **kwargs):
        # در این نسخه فقط بازگشت self کافی‌ست
        return self

    # ----------------------------------------------------------
    # 🪪 متادیتا
    # ----------------------------------------------------------
    @property
    def _llm_type(self):
        return "google_gemini"


# ----------------------------------------------------------
# 🔍 تست مستقیم مدل
# ----------------------------------------------------------
def get_google_llm(model_name="gemini-1.5-flash"):
    print(f"✅ Connected to Google Gemini ({model_name})")
    return GoogleGeminiLLM(model_name=model_name)


if __name__ == "__main__":
    llm = get_google_llm()
    print("🔍 Test prompt →", llm.predict("Say hello from Gemini 1.5 Flash!"))
