# RAG Telecom Chatbot

A Retrieval-Augmented Generation (RAG) customer care chatbot for telecom support. It answers questions about mobile connectivity, billing, SIM issues, and roaming by retrieving relevant context from three knowledge sources and generating responses with Qwen3-32B via Groq.

## Architecture

```
User question
     │
     ▼
Merged Retriever (top-k from each store)
  ├── ChromaDB · faq        (FAQ entries from CSV)
  ├── ChromaDB · tickets    (resolved support tickets from SQLite)
  └── ChromaDB · guides     (PDF guide chunks)
     │
     ▼
ChatPromptTemplate → Qwen3-32B (Groq) → Answer
```

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (runs locally via HuggingFace)  
**LLM:** `qwen/qwen3-32b` served by [Groq](https://groq.com)

## Project Structure

```
rag-telecom-chatbot/
├── app.py              # Streamlit web UI
├── main.py             # CLI entry point
├── rag_chain.py        # Builds the LangChain RAG chain
├── retriever.py        # Merges the three Chroma retrievers
├── ingest_faq.py       # Loads data/faq.csv → Chroma 'faq' collection
├── ingest_tickets.py   # Loads data/tickets.db → Chroma 'tickets' collection
├── ingest_pdf.py       # Loads data/telecom_guide.pdf → Chroma 'guides' collection
├── explore_chroma.py   # Streamlit GUI to browse & visualize the vector store
├── data/
│   ├── faq.csv             # FAQ question/answer pairs
│   ├── tickets.db          # SQLite database of resolved support tickets
│   ├── telecom_guide.pdf   # Telecom user guide (chunked at ingest)
│   ├── seed_tickets.py     # Script to seed the tickets database
│   └── generate_pdf.py     # Script to generate the telecom guide PDF
├── chroma_store/       # Persisted Chroma vector database (created at ingest)
├── pyproject.toml
├── uv.lock
└── .env.example
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A [Groq API key](https://console.groq.com)
- A [HuggingFace token](https://huggingface.co/settings/tokens) (for downloading the embedding model)

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd rag-telecom-chatbot
uv sync          # or: pip install -e .
```

`uv sync` creates a virtual environment in `.venv/` with all dependencies.

> **⚠️ Activate the virtual environment before running anything.** This is the
> single most common source of errors (see [Troubleshooting](#troubleshooting)).
>
> ```bash
> source .venv/bin/activate     # your prompt should now show (.venv)
> ```
>
> Once activated, use `python` (not `python3`). If you skip activation, `python3`
> falls back to your system Python, which does **not** have this project's
> dependencies — you'll get `ModuleNotFoundError`. To run a script without
> activating, call the venv's Python directly: `.venv/bin/python <script>.py`.

**2. Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

> **🔒 Put real keys only in `.env`, never in `.env.example`.** `.env.example` is a
> committed template and must contain placeholder values only. `.env` is
> gitignored and holds your actual secrets. Committing a real key will get your
> push rejected by GitHub's secret-scanning push protection — and if a key is
> ever exposed, **revoke and regenerate it** ([Groq](https://console.groq.com/keys),
> [HuggingFace](https://huggingface.co/settings/tokens)).

**3. Ingest data into Chroma**

Run the three ingestion scripts once to build the vector store (make sure the
venv is activated first — see the warning under step 1):

```bash
python ingest_faq.py
python ingest_tickets.py
python ingest_pdf.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB); setting
`HF_TOKEN` avoids rate limits but isn't strictly required.

Each script embeds the source data and persists it to `chroma_store/`. Re-run a script only when its source data changes.

## Running the App

**Streamlit web UI**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar has one-click sample questions and a button to clear the conversation history.

**CLI**

```bash
python main.py
```

Interactive prompt — type a question and press Enter. Type `quit` to exit.

## Exploring the Vector Store

To inspect what got embedded — and to *see* how the embeddings cluster — launch
the explorer GUI:

```bash
streamlit run explore_chroma.py        # venv activated
# or: .venv/bin/python -m streamlit run explore_chroma.py
```

- **Browse tab** — every document and its metadata, per collection, as a table.
- **2D map tab** — the 384-dimensional embeddings reduced to 2D (PCA or t-SNE)
  and plotted, colored by collection. Points that sit close together are
  semantically similar — this is exactly what the retriever exploits.

Uses only already-installed libraries (streamlit, chromadb, scikit-learn); no
extra setup. Run the ingestion scripts first so there's data to show.

## Data Sources

| Collection | Source file | Granularity |
|---|---|---|
| `faq` | `data/faq.csv` | 1 document per FAQ row |
| `tickets` | `data/tickets.db` | 1 document per resolved ticket |
| `guides` | `data/telecom_guide.pdf` | Chunks of 600 chars with 100-char overlap |

The retriever fetches the top 3 results from each collection (9 context documents total) for every query.

## Regenerating Seed Data

```bash
# Seed the SQLite ticket database
python data/seed_tickets.py

# Regenerate the PDF guide
python data/generate_pdf.py
```

After regenerating, re-run the corresponding ingest script.

## Troubleshooting

Real issues hit while setting this project up, and their fixes.

### `ModuleNotFoundError: No module named 'pandas'` (or langchain, chromadb, …)

Your virtual environment isn't active, so the script ran under system Python.
The dependencies live in `.venv/`, not in system Python.

```bash
source .venv/bin/activate     # then re-run with `python ...`
# or run without activating:
.venv/bin/python ingest_faq.py
```

### `zsh: command not found: python`

macOS has no bare `python` command, and an **inactive** venv doesn't provide one
either — that's the tell-tale sign your venv is not activated. Either activate it
(`source .venv/bin/activate`, after which `python` works) or use `python3` /
`.venv/bin/python` explicitly.

### How do I know the venv is actually active?

Your shell prompt shows `(.venv)`, and:

```bash
echo $VIRTUAL_ENV     # prints .../rag-telecom-chatbot/.venv
which python          # points into .venv/bin
```

If `$VIRTUAL_ENV` is empty, activation didn't take — re-run `source .venv/bin/activate`.

### `git push` rejected: "Push cannot contain secrets" (GH013)

A real API key was committed (usually in `.env.example`). GitHub's push
protection blocks it. Fix:

1. Move the real key into `.env` (gitignored); restore placeholder values in
   `.env.example`.
2. Remove the secret from history. If it's only in the latest commit:
   `git commit --amend` after fixing the file. If it's deeper, rewrite history
   (`git rebase` / `git filter-repo`).
3. **Revoke and regenerate the exposed key** — scrubbing history isn't enough
   once a secret has left your machine.

### `.env.example` keeps showing up in commits even though it's in `.gitignore`

`.gitignore` only affects **untracked** files. A file already committed stays
tracked and ignores gitignore rules. Untrack it with:

```bash
git rm --cached .env.example      # keeps the local file, stops tracking it
```

### `python3 ingest_pdf.py` — "No collections" / retriever finds nothing

You must run **all three** ingestion scripts before `app.py` / `main.py` have
anything to retrieve. Re-run any you skipped.
