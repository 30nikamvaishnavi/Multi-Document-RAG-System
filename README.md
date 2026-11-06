# Multi-Document Agentic RAG System

An end-to-end document Q&A app that answers questions across PDF, Word, PowerPoint,
Excel, text, Markdown, HTML, and CSV files. Citations include PDF page numbers,
PowerPoint slide numbers, spreadsheet names, or the source document. A LangGraph
agent routes general conversation without retrieval, rewrites document questions
for search, retrieves evidence from Chroma, and checks grounding.

Built with **Gemini**, **LangGraph**, **Chroma**, **Streamlit**, and **pypdf**.

## Features

- Ask questions across multiple supported document formats with source citations.
- Route general conversation without searching the document index.
- Rewrite document questions to improve retrieval.
- Check generated answers against retrieved evidence and retry when ungrounded.
- Upload PDFs, build the persistent index, and chat in Streamlit.

## Project structure

```
Multi-Document RAG System/
├── app.py              # Streamlit UI
├── ingestion.py        # Multi-format extraction and source-aware chunking
├── vectorstore.py      # Persistent Chroma wrapper
├── rag_engine.py       # Retrieval and citation-grounded generation
├── agentic_rag.py      # LangGraph routing and grounding loop
├── requirements.txt    # Dependencies
├── .env.example        # Environment variable template
├── docs/               # Uploaded PDFs
└── data/chroma/        # Persistent vector index
```

## Setup

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Get a Google API key from [Google AI Studio](https://aistudio.google.com/apikey),
   copy `.env.example` to `.env`, and set `GOOGLE_API_KEY`. The key is loaded
   from the environment and is not displayed in the app.

   On Windows:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload PDF, DOCX, TXT, MD, HTML, CSV, XLSX, or PPTX files and click **Save files**.
2. Click **Build Index** to extract text, chunk it, and create the Chroma index.
3. Ask document questions in the chat. Answers include source-location citations and a grounding status.
4. General questions are routed directly to Gemini and marked as not using document retrieval.

Scanned PDFs without an embedded text layer are not OCR-processed. PDF pages and
PowerPoint slides have numbered citations; Word/text/HTML/CSV files cite the
document, and Excel files cite the sheet name.

## Agent flow

```
START → route → (direct_answer | rewrite → retrieve → generate → ground_check → (retry → rewrite | END))
```

- `route` chooses whether the message needs evidence from uploaded documents.
- `rewrite` rewrites document questions for better retrieval.
- `retrieve` fetches relevant chunks from Chroma.
- `generate` answers using only the retrieved context and cites document pages.
- `ground_check` checks the answer against retrieved context and triggers a retry when needed.
- `direct_answer` handles general messages without searching the document index.

The retry count is configurable in `app.py` when constructing the agent. The
Gemini model and prompts are configured in `agentic_rag.py`; the embedding model
and collection name are configured in `vectorstore.py`.

## Tests

Run the offline unit tests (no Gemini API key or network access required):

```bash
python -m unittest test_core
```

The optional `test_e2e.py` script generates sample PDFs and can also run an
end-to-end Gemini query when `GOOGLE_API_KEY` is configured.

## License

MIT
