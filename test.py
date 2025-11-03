import os

import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# ✅ Configure the free Google AI Studio key
genai.configure(api_key=api_key)

# ✅ Explicitly use v1 model naming (AI Studio free endpoint)
model = genai.GenerativeModel("models/gemini-1.5-flash")

prompt = "Write a haiku about finite element analysis in civil engineering."
response = model.generate_content(prompt)

print(response.text)
