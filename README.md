
# ClaimWise

An Indian health insurance policy intelligence and claim preparation assistant.

## Project goals

- Explain insurance policy clauses in plain English.
- Retrieve evidence using hybrid RAG.
- Generate source-grounded answers with citations.
- Prepare policy-based claim document checklists.
- Identify missing information and conflicting policy versions.
- Keep documents, indexes, and metadata locally.

## Technology

- Python 3.10
- Streamlit
- PyMuPDF and Tesseract OCR
- Ollama embeddings
- Qdrant
- BM25 and Reciprocal Rank Fusion
- LangGraph
- Groq API
- SQLite

## Privacy

Personal policy documents are stored locally.
Selected evidence may be sent to Groq for generation
only with the user's permission.

## Project status

Currently in the initial development setup phase.
