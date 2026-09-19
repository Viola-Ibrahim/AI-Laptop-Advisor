# Labi — Multilingual RAG Laptop Advisor

A multilingual **Retrieval-Augmented Generation (RAG)** assistant that helps users make laptop decisions using information retrieved from a curated knowledge base.

Labi supports **Arabic, English, and Franco Arabic**, maintains conversation context, performs bilingual retrieval, and verifies generated factual claims against retrieved sources before showing the answer.

If the retrieved context does not support an answer, Labi refuses to guess instead of presenting unsupported information.

---

##  Core Features

*  **Multilingual** — supports Arabic, English, Franco Arabic, and mixed-language queries.
*  **Bilingual Retrieval** — searches using both the original query and its translated form to retrieve relevant sources across languages.
*  **Intent-Aware Retrieval** — classifies queries such as recommendations, brand-specific questions, comparisons, general information, and price checks, then re-ranks results using intent and source metadata.
*  **Grounding Verification** — checks generated factual claims against retrieved context using semantic similarity and rejects unsupported responses.
*  **Conversation Memory** — maintains chat history to handle follow-up questions naturally.
*  **Runtime Knowledge Ingestion** — supports adding PDF, CSV, and publicly accessible Google Sheets sources during a session.
*  **Streamlit Interface** — exposes the assistant's detected language, intent, grounding status, and retrieved sources.

---

##  Architecture

### Indexing

```text
Source Documents
      ↓
Extraction & Cleaning
      ↓
Chunking
      ↓
Multilingual Embeddings
      ↓
ChromaDB
```

### Query Time

```text
User Query
    ↓
Language Detection
    ↓
Intent Classification
    ↓
Bilingual Retrieval
    ↓
Intent-Based Re-ranking
    ↓
Response Generation
    ↓
Grounding Verification
    ↓
┌───────────────┬──────────────────┐
│     Pass      │       Fail       │
│       ↓       │        ↓         │
│ Show Response │  Refuse / No-Answer
└───────────────┴──────────────────┘
```

---

##  Tech Stack

| Layer          | Technology                              |
| -------------- | --------------------------------------- |
| LLM            | Groq — `openai/gpt-oss-120b`            |
| Embeddings     | `paraphrase-multilingual-mpnet-base-v2` |
| Vector Store   | ChromaDB                                |
| Orchestration  | LangChain (LCEL)                        |
| Interface      | Streamlit                               |
| PDF Extraction | PyMuPDF                                 |

---

##  Project Structure

```text
AI-Laptop-Advisor/
│
├── app.py
├── requirements.txt
├── .env
│
├── manuals/
│   ├── processed/
│   └── file_metadata.json
│
├── data/
│   └── chroma_db/
│
└── notebooks/
    └── full_project_notebook.ipynb
```

---

##  Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Create a `.env` file:

```env
GROQ_API_KEY=your_key_here
```

### 4. Build the knowledge base

Run the notebook from top to bottom:

```text
notebooks/full_project_notebook.ipynb
```

### 5. Start the application

```bash
streamlit run app.py
```

---

##  Limitations

* The current knowledge base contains a limited set of sources, so queries outside its coverage may receive a no-answer response.
* Some Arabic PDFs may produce reversed or incorrectly ordered text during extraction because of RTL rendering in the original files.
* Google Sheets must be publicly viewable to be ingested.
* Current evaluation relies on manual test cases rather than a comprehensive benchmark.

Expanding and improving the source corpus is the most direct way to improve knowledge coverage and answer quality.

---

##  Future Improvements

* Expand the laptop knowledge base with more models and sources.
* Improve Arabic RTL document extraction.
* Add a larger automated evaluation dataset.
* Improve claim-level grounding and source citations.
* Add more structured product data sources.
