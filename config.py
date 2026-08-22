import os
from dotenv import load_dotenv

load_dotenv()

# The GROQ_API_KEY is now loaded securely from .env
CSV_PATH = "cleaned_reviews.csv"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "feedback_forge"

# Groq model ids (llama-3.x models were retired from Groq's API)
GROQ_FAST_MODEL = "openai/gpt-oss-20b"
GROQ_STRATEGIST_MODEL = "openai/gpt-oss-120b"
