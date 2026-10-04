# DataAnalyst AI

A local analytics workspace that answers questions from your **spreadsheets and documents together**: exact numbers from Pandas, explanations retrieved from PDFs, and every claim tied to a file and page.

## What I Learned

- Building RAG as a pipeline of separable stages (load, chunk, embed, store, retrieve, rerank, generate) so each can be tested and swapped.
- Why arithmetic belongs in Pandas/SQL and not in an LLM, and how to hand the model only the evidence it needs.
- Hybrid retrieval: combining vector similarity with BM25 using Reciprocal Rank Fusion, then reranking with a cross-encoder.
- Building citations from retrieval metadata instead of trusting model output.
- Running a local OpenAI-compatible LLM (LM Studio) with graceful behaviour when it is offline.

## Problem

"Chat with PDF" tools cannot answer business questions that need both numbers and context, such as *which product declined most, and what did management say about it?* Pure LLM answers also tend to guess arithmetic and invent citations.

## Solution

Questions are routed to document retrieval (RAG), structured analytics, or both (HYBRID). Pandas computes results exactly; the local Qwen model only summarises retrieved evidence. Citations come from stored metadata. Documents stay on your machine because the LLM is hosted locally.

## Key Features

- PDF/TXT/Markdown ingestion with page-level metadata, persisted in ChromaDB (no duplicate chunks; re-index and delete supported)
- Hybrid retrieval: vector + BM25, Reciprocal Rank Fusion, local CrossEncoder reranker, relevance threshold
- CSV/Excel analytics: totals, averages, counts, group-bys, rankings, monthly trends, period-over-period change, percentage change
- Rule-based router (RAG / ANALYTICS / HYBRID) with local-LLM tie-breaking for ambiguous questions
- HYBRID answers split into **Calculated** and **Document evidence** with sources
- Explicit multi-step analysis (e.g. weakest region, then its best-selling product, then document search)
- Conversational follow-ups ("Why?") via query rewriting and a bounded history window
- Plotly charts only when they help; "insufficient evidence" answers instead of forced ones
- Local evaluation: retrieval hit rate, citation accuracy, answer relevance, grounding, latency
- Optional PostgreSQL query history (the app runs fine without it)

## Architecture

```
Browser (HTML/CSS/JS + Plotly)
        |  REST
FastAPI (backend/main.py)
        |
AnalystPipeline (backend/pipeline.py)
   |-- QuestionRouter ------------- rules, then LM Studio if ambiguous
   |-- AnalyticsEngine ------------ Pandas (+ read-only SQL fallback)
   |-- HybridRetriever ------------ Chroma vectors + BM25 -> RRF -> reranker
   '-- LMStudioClient (generator) - http://localhost:1234/v1
```

## How It Works

1. Upload files. PDFs are chunked (800 chars, 150 overlap), embedded locally with all-MiniLM-L6-v2 and stored in `chroma_db/`. CSV/Excel files are loaded into Pandas.
2. Ask a question. Follow-ups are rewritten into standalone questions (the original is preserved).
3. The router picks RAG, ANALYTICS or HYBRID.
4. Results are returned with a calculation trace, sources, optional chart and retrieved passages.

## Tech Stack

Python 3.11, FastAPI, Uvicorn, PyMuPDF, Pandas, OpenPyXL, sentence-transformers, ChromaDB, rank-bm25, LM Studio (Qwen2.5-7B-Instruct-1M), optional PostgreSQL (psycopg), vanilla JS, Plotly.js.

## Project Structure

```
dataanalyst-ai/
  backend/
    main.py  router.py  config.py  pipeline.py  services.py  documents.py  db.py
    rag/         loader chunker embeddings retriever bm25_retriever hybrid_retriever
                 reranker generator query_rewriter conversation
    analytics/   csv_analyzer excel_analyzer sql_engine engine chart_generator
    evaluation/  evaluator
  frontend/      index.html style.css dashboard.js
  tests/         loader, chunker, embeddings, retrieval, analytics, api
  evaluation_data/  eval_dataset.example.json
  uploads/  chroma_db/  .env.example  requirements.txt
```

## Local Setup

```powershell
cd C:\Users\ADMIN\OneDrive\Desktop\RAG\dataanalyst-ai
python -m venv venv          # skip if it already exists
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env       # skip if .env already exists
```

The first start downloads two small models from Hugging Face (embedding model and reranker). After that everything runs offline. If the reranker cannot be downloaded the app falls back to fusion order and says so in `/api/health`.

## LM Studio Setup

1. Load **Qwen2.5-7B-Instruct-1M** in LM Studio.
2. Open the *Developer* / *Local Server* tab and start the server (default `http://localhost:1234`).
3. The app calls `GET /v1/models` and uses the real model identifier LM Studio reports, so `LM_STUDIO_MODEL` only needs to be roughly right. Check `GET /api/model-status` to see what was detected.

If the server is not running the backend stays up. Upload and retrieval still work, and answers explain that LM Studio is not reachable.

## Running Backend

```powershell
python -m uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000. Run tests with `python -m pytest`. The embedding test is skipped automatically when the model cannot be loaded.

## Using the Application

1. Upload a PDF report plus a CSV/Excel sheet (needs a date column, a numeric metric such as Revenue, and optionally Product/Region columns).
2. Ask, for example:
   - "What was total revenue in August?" (ANALYTICS)
   - "What does the August report say about inventory?" (RAG)
   - "Which product had the largest revenue decline and what explanation was given in management reports?" (HYBRID)
   - then "Why?" as a follow-up
3. Toggle **Debug** to see the chosen route, rewritten question and retrieval details.

## RAG Pipeline

question → (rewrite) → vector search + BM25 → Reciprocal Rank Fusion → relevance threshold → CrossEncoder rerank (20 → 5) → Qwen with numbered evidence → citation validation → answer + sources.

Citations are built from chunk metadata. Invented `[S#]` ids are stripped, and if the evidence is weak the app answers: *"I couldn't find enough evidence in the uploaded documents to answer this reliably."*

## Hybrid Analytics Pipeline

1. Pandas finds, for instance, the product with the largest decline (absolute change, with % change reported).
2. Code builds a retrieval query from that result (for example "Product B Revenue decline reason explanation August").
3. Evidence is retrieved and reranked.
4. Qwen summarises only the document passages. The calculated text is inserted by code, so the model cannot change a number.
5. Response contains CALCULATED, DOCUMENT EVIDENCE, sources, chart, and the calculation trace.

## Evaluation

Create `evaluation_data/eval_dataset.json` (see the example file):

```json
[{"question": "...", "expected_answer": "...", "expected_file": "Management_Report.pdf", "expected_page": 12}]
```

Then `POST /api/evaluate` (empty JSON body `{}` uses that file, or pass `{"dataset": [...], "generate": false}` for retrieval-only). Reports are saved in `evaluation_results/`. Metrics: retrieval hit rate, citation accuracy, answer relevance (term recall against the expected answer), faithfulness (share of answer terms found in the retrieved context) and latency. Relevance and faithfulness are simple lexical heuristics, not a replacement for human review.

## Known Limitations

- No OCR: scanned PDFs without a text layer are rejected as unreadable.
- Rule-based analytics cover common questions; anything else falls back to an LLM-written read-only SQL query that SQLite executes.
- Period comparisons use calendar months. Without a month in the question, "decline" compares the latest two months and "growth" compares first to last; the answer states which periods were used.
- Conversation memory is in-process and resets when the server restarts.
- Plotly.js is loaded from a CDN, so charts need internet access unless you vendor the file.

## Future Improvements

- OCR for scanned PDFs, token-aware or semantic chunking
- Quarterly/yearly periods and multi-table joins
- Persistent conversation history in PostgreSQL
- Streaming answers and a richer evaluation set with LLM-free faithfulness checks
