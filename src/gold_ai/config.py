import os

from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment.")


if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the environment.")
