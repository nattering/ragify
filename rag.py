import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import openai

load_dotenv()

MAX_FILE_SIZE_MB = 20
MIN_EXTRACTED_CHARS = 200  # below this, the PDF is probably scanned/image-only
MAX_HISTORY_TURNS = 4      # how many past question/answer pairs to keep as context


class RagifyError(Exception):
    """Base error for anything that goes wrong in the pipeline, with a message safe to show the user."""
    pass


class EmptyDocumentError(RagifyError):
    pass


class FileTooLargeError(RagifyError):
    pass


class ApiKeyError(RagifyError):
    pass


class ApiRateLimitError(RagifyError):
    pass


class ApiConnectionErrorRagify(RagifyError):
    pass


CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the follow-up question as a standalone question that includes any "
     "necessary context from the conversation history below. If the follow-up "
     "question is already standalone, return it unchanged. Output only the "
     "rewritten question, nothing else — no preamble, no quotes."),
    ("human", "Conversation history:\n{history}\n\nFollow-up question: {question}\n\nStandalone question:")
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering questions about a document, using only "
     "the context passages provided below. If the answer isn't in the context, say "
     "you don't know based on the document rather than guessing. The conversation "
     "history is there to help you understand what's being asked, not as a source "
     "of facts.\n\nContext:\n{context}"),
    ("human", "{question}")
])


def validate_file_size(file_path: str):
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise FileTooLargeError(
            f"That file is {size_mb:.1f}MB — Ragify currently handles PDFs up to {MAX_FILE_SIZE_MB}MB."
        )


def build_vectorstore(pdf_path: str):
    """Load a PDF, split it into chunks, and build a FAISS vector store."""
    validate_file_size(pdf_path)

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        raise RagifyError(f"Couldn't open that PDF — it may be corrupted or password-protected. ({e})")

    total_text = "".join(doc.page_content for doc in documents).strip()
    if len(total_text) < MIN_EXTRACTED_CHARS:
        raise EmptyDocumentError(
            "No readable text was found in this PDF. It's likely a scanned document or "
            "image-only file — Ragify needs a PDF with selectable text, not a photo of pages."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except openai.AuthenticationError:
        raise ApiKeyError(
            "Your OpenAI API key was rejected. Check that OPENAI_API_KEY in your .env file is correct."
        )
    except openai.RateLimitError:
        raise ApiRateLimitError(
            "OpenAI rate limit or quota reached. Check your usage/billing on platform.openai.com and try again shortly."
        )
    except openai.APIConnectionError:
        raise ApiConnectionErrorRagify("Couldn't reach OpenAI's servers. Check your internet connection and try again.")

    return vectorstore


def build_qa_chain(vectorstore):
    """Return the pieces needed to answer questions: an LLM and a retriever."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return {"llm": llm, "retriever": retriever}


def _format_history(chat_history: list) -> str:
    """chat_history: list of {'question': str, 'answer': str} dicts, oldest first."""
    if not chat_history:
        return "(no earlier conversation)"
    recent = chat_history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent:
        lines.append(f"Q: {turn['question']}")
        lines.append(f"A: {turn['answer']}")
    return "\n".join(lines)


def _condense_question(llm, question: str, chat_history: list) -> str:
    if not chat_history:
        return question
    history_text = _format_history(chat_history)
    chain = CONDENSE_PROMPT | llm
    result = chain.invoke({"history": history_text, "question": question})
    standalone = result.content.strip()
    return standalone if standalone else question


def answer_question(qa_chain, question: str, chat_history: list = None):
    """
    Run a question through the pipeline, using chat_history for context on follow-ups.
    chat_history: list of {'question': str, 'answer': str} dicts, oldest first.
    Returns dict with 'answer', 'sources', and 'standalone_question' (what was actually searched for).
    """
    if not question or not question.strip():
        raise RagifyError("Ask something first — the question was empty.")

    chat_history = chat_history or []
    llm = qa_chain["llm"]
    retriever = qa_chain["retriever"]

    try:
        standalone_question = _condense_question(llm, question, chat_history)
        docs = retriever.invoke(standalone_question)

        context = "\n\n".join(doc.page_content for doc in docs)
        answer_chain = ANSWER_PROMPT | llm
        result = answer_chain.invoke({"context": context, "question": question})

    except openai.AuthenticationError:
        raise ApiKeyError("Your OpenAI API key was rejected. Check OPENAI_API_KEY in your .env file.")
    except openai.RateLimitError:
        raise ApiRateLimitError("OpenAI rate limit or quota reached. Try again in a moment.")
    except openai.APIConnectionError:
        raise ApiConnectionErrorRagify("Couldn't reach OpenAI's servers. Check your internet connection.")
    except Exception as e:
        raise RagifyError(f"Something went wrong answering that question: {e}")

    return {
        "answer": result.content,
        "sources": docs,
        "standalone_question": standalone_question
    }