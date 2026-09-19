"""
Labi — Egyptian Laptop Advisor (Streamlit interface)

Run from the project root (same folder as .env, manuals/, data/):
    streamlit run app.py

Requires: streamlit, python-dotenv, chromadb, sentence-transformers,
langchain-core, langchain-chroma, langchain-groq, scikit-learn, pypdf, pandas
"""

import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

load_dotenv(".env")

# ── Paths (project root layout) ──────────────────────────────────────────
CHROMA_PATH = os.path.join("data", "chroma_db")
METADATA_PATH = os.path.join("manuals", "file_metadata.json")


# ── Cached resources (loaded once per session, not on every rerun) ──────
@st.cache_resource(show_spinner="wait please...")
def load_resources():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        file_metadata = json.load(f)

    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    class SentenceTransformerEmbeddings(Embeddings):
        def __init__(self, m):
            self.model = m

        def embed_documents(self, texts):
            return self.model.encode(texts).tolist()

        def embed_query(self, text):
            return self.model.encode([text])[0].tolist()

    lc_embeddings = SentenceTransformerEmbeddings(model)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    vectorstore = Chroma(
        client=client,
        collection_name="laptop_kb",
        embedding_function=lc_embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

    return file_metadata, lc_embeddings, vectorstore, retriever, llm


file_metadata, lc_embeddings, vectorstore, retriever, llm = load_resources()

# ── Language detection ───────────────────────────────────────────────────
AR = re.compile(r"[\u0600-\u06FF]")
LAT = re.compile(r"[A-Za-z]")
FRANCO = re.compile(r"\b[a-z]{0,4}[23579][a-z]{1,}\b", re.I)


def detect_language(text):
    ar, lat = len(AR.findall(text)), len(LAT.findall(text))
    if ar == 0:
        return "franco" if FRANCO.search(text) else "en"
    if lat == 0:
        return "ar"
    return "ar" if lat / (ar + lat) < 0.4 else "mixed"


LANG_RULES = {
    "ar": "Respond in Egyptian Arabic. Keep technical terms and model names in English as written (RTX 4060, SSD, Core i7).",
    "en": "Respond in English, same persona and tone.",
    "mixed": "Respond in Egyptian Arabic, keeping English technical terms exactly as the user wrote them.",
    "franco": "The user is writing Franco Arabic (Arabic in Latin letters). Respond in normal Egyptian Arabic script — don't mimic their writing style.",
}

NO_ANSWER_PHRASE = "I don't have this information right now"


def is_no_answer(answer):
    return NO_ANSWER_PHRASE in answer


def translate_query(q, target):
    result = llm.invoke(
        f"Translate this search query to {target}. Output only the translation:\n{q}"
    ).content
    if isinstance(result, list):
        result = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in result
        )
    return result.strip()


def bilingual_retrieve(question, lang, k=4):
    docs = retriever.invoke(question)
    target = "English" if lang in ("ar", "franco") else "Arabic"
    try:
        docs += retriever.invoke(translate_query(question, target))
    except Exception:
        pass
    seen, out = set(), []
    for d in docs:
        key = d.page_content[:120]
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out[: k * 2]


def is_clarifying_question(answer):
    stripped = answer.strip()
    return (stripped.endswith("؟") or stripped.endswith("?")) and len(stripped) < 400


# ── Grounding check ───────────────────────────────────────────────────────
def split_sentences(text):
    sentences = re.split(r"[.!؟?\n]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def contains_factual_claim(sentence):
    return bool(re.search(r"\d", sentence)) or any(
        b in sentence for b in ["Dell", "HP", "Lenovo", "Apple", "RTX", "Core i", "SSD", "RAM"]
    )


def looks_grounded(answer, context, threshold=0.60):
    if is_no_answer(answer):
        return True, "no_answer", 1.0

    answer_sentences = [s for s in split_sentences(answer) if contains_factual_claim(s)]
    context_sentences = split_sentences(context)

    if not answer_sentences:
        return True, "no_factual_claims", 1.0
    if not context_sentences:
        return False, "no context retrieved", 0.0

    answer_vecs = lc_embeddings.embed_documents(answer_sentences)
    context_vecs = lc_embeddings.embed_documents(context_sentences)
    sims = cosine_similarity(answer_vecs, context_vecs)
    avg_similarity = float(np.mean(sims.max(axis=1)))

    if avg_similarity >= threshold:
        return True, "ok", avg_similarity
    return False, f"low similarity ({avg_similarity:.2f})", avg_similarity


# ── Intent classification ────────────────────────────────────────────────
INTENT_PROMPT = """Classify the user's message into ONE intent. Reply with ONLY valid JSON, nothing else.

Intents:
- "vague_request": user wants a laptop recommendation but hasn't given budget AND use case yet
- "ready_to_recommend": user wants a recommendation and has given budget and/or use case
- "brand_specific": user is asking about a specific brand's lineup/models (Dell, HP, Lenovo, Apple)
- "brand_comparison": user wants to compare two or more brands
- "general_info": general question about specs, terminology, or buying advice, not asking for a specific product
- "price_check": asking about price ranges only, not asking for a specific recommendation

Conversation so far:
{history}

User's latest message: {question}

JSON format: {{"intent": "...", "brand": "brand name or null", "needs_clarification": true/false}}"""


def classify_intent(question, chat_history):
    history_text = "\n".join(f"{role}: {msg}" for role, msg in chat_history[-6:])
    raw = llm.invoke(INTENT_PROMPT.format(history=history_text, question=question)).content
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "general_info", "brand": None, "needs_clarification": False}


def filtered_retrieve(question, lang, intent_data, k=4):
    docs = bilingual_retrieve(question, lang, k=k * 2)
    intent = intent_data.get("intent")
    brand = intent_data.get("brand")

    def score(doc):
        meta = file_metadata.get(doc.metadata.get("source_file", ""), {})
        s = 0
        if intent == "brand_specific" and brand and meta.get("brand", "").lower() == (brand or "").lower():
            s += 2
        if intent == "brand_comparison" and meta.get("content_type") == "brand_comparison":
            s += 2
        if intent in ("ready_to_recommend", "price_check") and meta.get("content_type") == "buying_advice":
            s += 1
        return s

    docs.sort(key=score, reverse=True)
    return docs[:k]


# ── Persona and main chain ───────────────────────────────────────────────
PERSONA_CORE = """You are "Labi" — an Egyptian laptop advisor with years of experience in the field.

Style:
- Keep it simple and conversational, no unnecessary technical jargon
- If a technical term matters, mention it and explain it briefly
- Be honest: if something isn't worth the money, say so
- Keep answers short and focused, not essays

Conversation flow (important):
- Before recommending any specific laptop, you need to know at minimum: budget, and main use case (gaming, study, design/editing, general use).
- If the user's message doesn't give you enough of this, do NOT recommend anything yet. Instead, ask ONE or TWO short follow-up questions to fill the gap — never a long list of questions at once.
- Check the conversation history: if the user already answered something earlier, don't ask it again.
- Once you have budget + use case, give a direct recommendation grounded in the context below.
- If the user asks a general/factual question, just answer it directly.

Accuracy rules (critical, never break these):
- Answer ONLY using the information in the "Available Context" below. Do not use your own general knowledge about laptops or prices.
- If the context does not contain the answer, reply with EXACTLY this phrase: "المعلومة دي مش موجودة عندي دلوقتي"
- Never invent prices or specs that are not explicitly stated in the context."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PERSONA_CORE + "\n\nتعليمات اللغة:\n{lang_rule}\n\nالسياق المتاح:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)
chain = prompt | llm | StrOutputParser()


def ask(question, chat_history):
    lang = detect_language(question)
    intent_data = classify_intent(question, chat_history)
    docs = filtered_retrieve(question, lang, intent_data)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    answer = chain.invoke(
        {
            "lang_rule": LANG_RULES[lang],
            "context": context,
            "chat_history": chat_history,
            "question": question,
        }
    )

    if intent_data.get("needs_clarification") and is_clarifying_question(answer):
        grounded, reason, score = True, "clarifying_question", None
    else:
        grounded, reason, score = looks_grounded(answer, context)
        if not grounded:
            answer = NO_ANSWER_PHRASE 

    sources = sorted({d.metadata.get("source_file", "?") for d in docs}) if not is_no_answer(answer) else []
    return {
        "answer": answer,
        "lang": lang,
        "intent": intent_data.get("intent"),
        "grounded": grounded,
        "score": score,
        "sources": sources,
    }


# ── User file upload helpers ──────────────────────────────────────────────
def extract_pdf_text(file_obj):
    reader = PdfReader(file_obj)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_csv_text(file_obj):
    df = pd.read_csv(file_obj)
    rows = df.apply(lambda r: ", ".join(f"{c}: {v}" for c, v in r.items()), axis=1)
    return "\n".join(rows)


def google_sheet_to_csv_url(sheet_url):
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not match:
        raise ValueError("please enter a valid sheet link")
    return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv"


def add_user_file_to_kb(splitter_source, source, source_type, display_name=None):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)

    if source_type == "pdf":
        text = extract_pdf_text(source)
        name = display_name or getattr(source, "name", "uploaded.pdf")
    elif source_type == "csv":
        text = extract_csv_text(source)
        name = display_name or getattr(source, "name", "uploaded.csv")
    elif source_type == "google_sheet":
        text = extract_csv_text(google_sheet_to_csv_url(source))
        name = display_name or "google_sheet_upload"
    else:
        raise ValueError("unsupported file datatype")
    if not text.strip():
        raise ValueError("value error")
    chunks = splitter.split_text(text)
    metadatas = [{"source_file": name, "content_type": "user_uploaded", "language": "unknown"} for _ in chunks]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas)
    return len(chunks), name


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Lapi - AI laptop Advisor", page_icon="💻", layout="centered")

st.markdown(
    """
    <style>
    .stChatMessage p, .stChatMessage li { unicode-bidi: plaintext; text-align: start; }
    .labi-header { display: flex; align-items: center; gap: 14px; padding: 6px 0 18px 0; }
    .labi-avatar {
        font-size: 40px; background: #1D9E75; color: white; border-radius: 50%;
        width: 60px; height: 60px; display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .labi-title { font-size: 22px; font-weight: 600; margin: 0; }
    .labi-subtitle { font-size: 14px; color: gray; margin: 0; }
    .labi-badge {
        display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px;
        margin-right: 4px; background: #eee; color: #333;
    }
    .labi-badge-ok { background: #E1F5EE; color: #0F6E56; }
    .labi-badge-warn { background: #FAEEDA; color: #854F0B; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="labi-header">
        <div class="labi-avatar">💻</div>
        <div>
            <p class="labi-title">Lapi - AI laptop Advisor</p>
            <p class="labi-subtitle">بيرشحلك لابتوب صح من غير لخبطة، بالعربي أو الإنجليزي أو الفرانكو</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### about Lapi 💬")
    st.write(
    "A smart assistant that recommends laptops based on real sources "
    "(articles, price tables, buying guides) — not guesses. "
    "It understands Arabic, English, and Franco Arabic, and remembers what you've told it."
)
    st.markdown("---")
    st.markdown("### Add your file")
    upload_type = st.radio("type", ["PDF", "CSV", "Google Sheet link"], horizontal=False)

    if upload_type in ("PDF", "CSV"):
        uploaded = st.file_uploader(
            "choose file", type=["pdf"] if upload_type == "PDF" else ["csv"]
        )
        if uploaded and st.button("add"):
            try:
                n, name = add_user_file_to_kb(
                    None, uploaded, "pdf" if upload_type == "PDF" else "csv"
                )
                st.success(f"Added Succefully")
            except Exception as e:
                st.error(f"error : {e}")
    else:
        sheet_url = st.text_input("Sheet link (make sure it's public)")
        if sheet_url and st.button("add"):
            try:
                n, name = add_user_file_to_kb(None, sheet_url, "google_sheet")
                st.success(f"Added succefully")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    if st.button(" Delete History 🗑️"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.messages:
    avatar = "💻" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            badge_class = "labi-badge-ok" if meta.get("grounded") else "labi-badge-warn"
            score_txt = f"{meta['score']:.2f}" if meta.get("score") is not None else "n/a"
            src_txt = ", ".join(meta.get("sources", [])[:3]) or "—"
            st.markdown(
                f'<span class="labi-badge">{meta.get("lang")}</span>'
                f'<span class="labi-badge">{meta.get("intent")}</span>'
                f'<span class="labi-badge {badge_class}">grounded: {meta.get("grounded")} ({score_txt})</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"resources: {src_txt}")

question = st.chat_input("Ask your question ... (عربي / English / franco)")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="💻"):
        with st.spinner("Lapi thinking ..."):
            result = ask(question, st.session_state.chat_history)
        st.markdown(result["answer"])
        badge_class = "labi-badge-ok" if result["grounded"] else "labi-badge-warn"
        score_txt = f"{result['score']:.2f}" if result["score"] is not None else "n/a"
        src_txt = ", ".join(result["sources"][:3]) or "—"
        st.markdown(
            f'<span class="labi-badge">{result["lang"]}</span>'
            f'<span class="labi-badge">{result["intent"]}</span>'
            f'<span class="labi-badge {badge_class}">grounded: {result["grounded"]} ({score_txt})</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"resources: {src_txt}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "meta": {
                "lang": result["lang"],
                "intent": result["intent"],
                "grounded": result["grounded"],
                "score": result["score"],
                "sources": result["sources"],
            },
        }
    )
    st.session_state.chat_history.extend([("human", question), ("ai", result["answer"])])