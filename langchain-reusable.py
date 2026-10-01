# Install once: pip install -U langchain-groq python-dotenv
from os import environ, getenv
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Read the API key from the .env beside this script.
env_file = Path(__file__).with_name(".env")
if env_file.is_file():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            environ.setdefault(key.strip(), value.strip().strip("\"'"))
if not getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is missing; add it to this project's .env file.")

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
)
prompt = ChatPromptTemplate.from_template(
    "Explain {Topic} in 5 simple points."
)
chain = prompt | llm | StrOutputParser()


def explain_topic(topic: str) -> str:
    return chain.invoke({"Topic": topic})
print(explain_topic("Docker"))
print(explain_topic("LangChain"))
print(explain_topic("Terraform"))
