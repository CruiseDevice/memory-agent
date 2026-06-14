import os

from openai import OpenAI

client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
