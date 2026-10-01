# Web research multi-agent demo

This project uses a Groq chat model with LangChain agents and Tavily web search.
The supervisor delegates work to three focused researchers:

- Primary-source researcher: official, government, academic, and original sources.
- Independent researcher: reporting and expert context.
- Claim checker: supporting evidence, counterevidence, and disagreements.

The researchers return search-backed findings with source URLs. The supervisor
combines them into an answer with links beside the relevant claims.

## Setup

1. Install the dependencies in the active virtual environment:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. If `.env` does not exist, copy `.env.example` to `.env`; otherwise preserve it
   and add `GROQ_API_KEY`. Tavily Search uses `TAVILY_API_KEY` when present;
   otherwise it uses Tavily's rate-limited keyless Search endpoint. Add a Tavily
   key from [tavily.com](https://tavily.com/) for authenticated access.
3. Run a research question:

   ```powershell
   .\.venv\Scripts\python.exe web_research.py "What changed in Python 3.14?"
   ```

Set `GROQ_MODEL` in `.env` to choose another Groq model that supports tool use.
The default is `qwen/qwen3.8-27b`.
