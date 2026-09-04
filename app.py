import os
import tempfile
import base64
import streamlit as st
from rag import build_vectorstore, build_qa_chain, answer_question, RagifyError

st.set_page_config(page_title="Ragify", page_icon="favicon-32.png", layout="centered")

# ---------- Logo ----------
with open("ragify_logo_dark.svg", "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Public+Sans:wght@400;500;600&display=swap');

:root {
    --ink: #16202B;
    --ink-soft: #1E2C3A;
    --ink-softer: #24333F;
    --parchment: #EDE3CC;
    --parchment-dim: #E1D4B0;
    --amber: #B8863D;
    --amber-bright: #CE9C4E;
    --sage: #6E8B74;
    --text-light: #EDE6D8;
    --text-dark: #2A2118;
    --danger: #B85C3D;
}

html, body, .stApp {
    background-color: var(--ink) !important;
    background-image:
        radial-gradient(ellipse 900px 500px at 15% -10%, rgba(184,134,61,0.10), transparent 60%),
        radial-gradient(ellipse 700px 500px at 100% 15%, rgba(110,139,116,0.08), transparent 55%),
        repeating-linear-gradient(135deg, rgba(237,227,204,0.012) 0px, rgba(237,227,204,0.012) 1px, transparent 1px, transparent 3px);
    color: var(--text-light);
    font-family: 'Public Sans', sans-serif;
}

[class*="st-"] { font-family: 'Public Sans', sans-serif; }

.block-container {
    max-width: 680px;
    padding-top: 3rem;
}

::selection { background: var(--amber); color: var(--ink); }

::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: var(--ink); }
::-webkit-scrollbar-thumb { background: var(--ink-softer); border-radius: 10px; border: 2px solid var(--ink); }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }

@keyframes plate-in {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes rule-draw {
    from { width: 0%; }
    to   { width: 100%; }
}
.ragify-plate {
    position: relative;
    padding: 1.3rem 0 1.1rem 0;
    margin-bottom: 0.5rem;
    text-align: center;
    animation: plate-in 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.ragify-plate::before, .ragify-plate::after {
    content: "";
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    height: 1px;
    background: var(--amber);
    animation: rule-draw 0.9s cubic-bezier(0.16, 1, 0.3, 1) 0.15s both;
}
.ragify-plate::before { top: 0; }
.ragify-plate::after { bottom: 0; }
.ragify-plate img {
    display: block;
    margin: 0 auto 0.5rem auto;
}
.ragify-plate h1 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.5rem;
    color: var(--parchment);
    letter-spacing: 0.02em;
    margin: 0;
}
.ragify-plate p {
    font-family: 'Public Sans', sans-serif;
    color: var(--sage);
    font-size: 0.95rem;
    margin: 0.35rem 0 0 0;
    font-style: italic;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: var(--ink-soft) !important;
    border: 1px dashed var(--amber) !important;
    border-radius: 3px !important;
    padding: 1.1rem 1.3rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 1.2rem !important;
    flex-wrap: nowrap !important;
    transition: border-color 0.25s ease, background-color 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--amber-bright) !important;
    background-color: var(--ink-softer) !important;
    box-shadow: 0 0 0 3px rgba(184,134,61,0.12);
}
[data-testid="stFileUploaderDropzone"] section {
    display: flex !important;
    align-items: center !important;
    gap: 1rem !important;
    flex: 1 !important;
}
[data-testid="stFileUploaderDropzone"] input[type="file"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 1px !important;
    height: 1px !important;
}
[data-testid="stFileUploaderDropzone"] svg { display: none !important; }
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p { color: var(--parchment) !important; }
[data-testid="stFileUploader"] section button {
    background-color: transparent !important;
    border: 1px solid var(--amber) !important;
    border-radius: 3px !important;
    padding: 0.5rem 1.2rem !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
    position: relative !important;
    min-width: 130px !important;
    min-height: 2.4rem !important;
    transition: transform 0.15s ease, background-color 0.15s ease;
}
[data-testid="stFileUploader"] section button * {
    visibility: hidden !important;
}
[data-testid="stFileUploader"] section button::after {
    content: "Browse files";
    visibility: visible !important;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: var(--amber-bright);
    white-space: nowrap;
    font-family: 'Public Sans', sans-serif;
    font-size: 0.9rem;
}
[data-testid="stFileUploader"] section button:hover {
    background-color: var(--amber) !important;
    transform: translateY(-1px);
}
[data-testid="stFileUploader"] section button:hover::after {
    color: var(--ink);
}

@keyframes msg-in {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.4rem 0 !important;
    animation: msg-in 0.35s ease both;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }

[data-testid="stChatMessageContent"] {
    border-radius: 3px !important;
    padding: 0.85rem 1.1rem !important;
    font-size: 0.96rem;
    line-height: 1.55;
    transition: box-shadow 0.2s ease;
}

.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background-color: var(--amber) !important;
    color: var(--ink) !important;
    margin-left: 15%;
    border-left: 3px solid var(--amber-bright);
}
.stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background-color: var(--parchment) !important;
    color: var(--text-dark) !important;
    margin-right: 8%;
    border-left: 3px solid var(--sage);
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
}

[data-testid="stChatInput"] textarea {
    background-color: var(--ink-soft) !important;
    color: var(--text-light) !important;
    border: 1px solid var(--amber) !important;
    border-radius: 3px !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--amber-bright) !important;
    box-shadow: 0 0 0 3px rgba(184,134,61,0.18) !important;
}
[data-testid="stChatInput"] button {
    color: var(--amber-bright) !important;
    transition: transform 0.15s ease;
}
[data-testid="stChatInput"] button:hover { transform: scale(1.1); }

.ragify-sources { margin: 0.4rem 0 1.2rem 0; padding-left: 1.1rem; border-left: 2px solid var(--sage); }
.ragify-sources summary {
    cursor: pointer;
    color: var(--sage);
    font-size: 0.82rem;
    font-style: italic;
    transition: color 0.15s ease;
}
.ragify-sources summary:hover { color: var(--amber-bright); }
.ragify-source-item {
    font-size: 0.82rem;
    color: var(--text-light);
    opacity: 0.75;
    margin-top: 0.4rem;
    line-height: 1.4;
}
.ragify-source-item .page-mark {
    color: var(--amber-bright);
    font-style: italic;
    margin-right: 0.4rem;
}

.ragify-rewrite {
    font-size: 0.78rem;
    color: var(--sage);
    font-style: italic;
    margin: 0.3rem 0 0.2rem 0;
    opacity: 0.85;
}

[data-testid="stAlert"] {
    background-color: var(--ink-soft) !important;
    color: var(--parchment) !important;
    border-left: 3px solid var(--amber) !important;
    border-radius: 3px !important;
    animation: msg-in 0.4s ease both;
}

.ragify-error {
    background-color: var(--ink-soft) !important;
    color: var(--parchment) !important;
    border-left: 3px solid var(--danger) !important;
    border-radius: 3px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.9rem;
    animation: msg-in 0.4s ease both;
}

[data-testid="stSpinner"] > div { color: var(--amber-bright) !important; }
[data-testid="stSpinner"] svg { fill: var(--amber-bright) !important; }
[data-testid="stSpinner"] p { color: var(--sage) !important; font-style: italic; }

@keyframes drift {
    0%, 100% { opacity: 0.65; transform: translateY(0); }
    50% { opacity: 1; transform: translateY(-3px); }
}
.ragify-empty {
    text-align: center;
    color: var(--sage);
    font-style: italic;
    margin-top: 2.5rem;
    animation: drift 3.2s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown(f"""
<div class="ragify-plate">
    <img src="data:image/svg+xml;base64,{logo_b64}" width="56">
    <h1>Ragify</h1>
    <p>Bring a document. Leave with its answers.</p>
</div>
""", unsafe_allow_html=True)

# ---------- State ----------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None


def show_error(message: str):
    st.markdown(f'<div class="ragify-error">{message}</div>', unsafe_allow_html=True)


def get_chat_history_pairs():
    """Turn session_state.messages into [{'question':..., 'answer':...}, ...] pairs, oldest first."""
    pairs = []
    pending_question = None
    for m in st.session_state.messages:
        if m["role"] == "user":
            pending_question = m["content"]
        elif m["role"] == "assistant" and pending_question is not None:
            pairs.append({"question": pending_question, "answer": m["content"]})
            pending_question = None
    return pairs


def looks_unanswered(answer_text: str) -> bool:
    """Heuristic: don't show 'Drawn from N passages' when the model said it doesn't know."""
    lowered = answer_text.lower()
    phrases = ["don't know", "do not know", "not mentioned", "no information", "not in the document", "not provided in the document"]
    return any(p in lowered for p in phrases)


# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded_name:
    st.session_state.qa_chain = None
    st.session_state.messages = []
    tmp_path = None

    try:
        with st.spinner("Reading and indexing your document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            vectorstore = build_vectorstore(tmp_path)
            st.session_state.qa_chain = build_qa_chain(vectorstore)
            st.session_state.last_uploaded_name = uploaded_file.name

        st.success(f"“{uploaded_file.name}” is indexed. Ask it something below.")

    except RagifyError as e:
        show_error(str(e))
        st.session_state.last_uploaded_name = None
    except Exception as e:
        show_error(f"Something unexpected went wrong while indexing this file: {e}")
        st.session_state.last_uploaded_name = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------- Chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") and not looks_unanswered(msg["content"]):
            rewrite_html = ""
            if msg.get("standalone_question") and msg["standalone_question"] != msg.get("original_question"):
                rewrite_html = f'<div class="ragify-rewrite">Searched as: “{msg["standalone_question"]}”</div>'
            sources_html = "".join(
                f'<div class="ragify-source-item"><span class="page-mark">p.{s["page"]}</span>{s["text"]}</div>'
                for s in msg["sources"]
            )
            st.markdown(f"""
            {rewrite_html}
            <details class="ragify-sources">
                <summary>Drawn from {len(msg["sources"])} passage(s)</summary>
                {sources_html}
            </details>
            """, unsafe_allow_html=True)

# ---------- Chat input ----------
if st.session_state.qa_chain is not None:
    question = st.chat_input("Ask something about your document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question, "sources": None})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                history = get_chat_history_pairs()
                with st.spinner("Reading through the pages..."):
                    result = answer_question(st.session_state.qa_chain, question, chat_history=history)
                    st.write(result["answer"])

                    sources = [
                        {"page": doc.metadata.get("page", "?"), "text": doc.page_content[:180].strip() + "…"}
                        for doc in result["sources"]
                    ]

                    if not looks_unanswered(result["answer"]):
                        rewrite_html = ""
                        if result["standalone_question"] != question:
                            rewrite_html = f'<div class="ragify-rewrite">Searched as: “{result["standalone_question"]}”</div>'

                        sources_html = "".join(
                            f'<div class="ragify-source-item"><span class="page-mark">p.{s["page"]}</span>{s["text"]}</div>'
                            for s in sources
                        )
                        st.markdown(f"""
                        {rewrite_html}
                        <details class="ragify-sources">
                            <summary>Drawn from {len(sources)} passage(s)</summary>
                            {sources_html}
                        </details>
                        """, unsafe_allow_html=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "original_question": question,
                    "standalone_question": result["standalone_question"]
                })

            except RagifyError as e:
                show_error(str(e))
                st.session_state.messages.append({"role": "assistant", "content": f"⚠ {e}", "sources": None})
            except Exception as e:
                msg = f"Something unexpected went wrong: {e}"
                show_error(msg)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠ {msg}", "sources": None})
else:
    st.markdown('<p class="ragify-empty">Upload a document above to open the reading room.</p>', unsafe_allow_html=True)