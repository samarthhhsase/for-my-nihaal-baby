# Install once: pip install -U langchain-groq python-dotenv
from pathlib import Path

from langchain_groq import ChatGroq
import os


# Load GROQ_API_KEY from the .env beside this script.
env_file = Path(__file__).with_name(".env")
if env_file.is_file():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key and not key.startswith("#"):
            os.environ.setdefault(key, value.strip().strip("\"'"))

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
)

response = llm.invoke("Explain Docker in simple terms.")
print(response.content)
