# Install once: pip install -U langchain langchain-groq python-dotenv
from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq


# Read the key from the .env beside this script; never hard-code it here.
load_dotenv(Path(__file__).with_name(".env"))
if not getenv("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is missing. Add GROQ_API_KEY=your_key to this project's .env file."
    )


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=ChatGroq(model="llama-3.3-70b-versatile"),
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content)
