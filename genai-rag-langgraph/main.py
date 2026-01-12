import os

# Disable SSL verification for HuggingFace + requests stack
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

class NoVerifyAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = 0
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)




import os
import ssl
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = ""

ssl._create_default_https_context = ssl._create_unverified_context
import time
import warnings
import logging
from typing import TypedDict




# -----------------------------
# 1. Environment hardening (local POC only)
# -----------------------------
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")

# -----------------------------
# 2. Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

# -----------------------------
# 3. FastAPI FIRST (CRITICAL)
# -----------------------------
from fastapi import FastAPI
from pydantic import BaseModel

api = FastAPI(title="Resilient RAG Service")

# -----------------------------
# 4. Request / Response models
# -----------------------------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

# -----------------------------
# 5. Lazy-loaded graph (GLOBAL)
# -----------------------------
graph_app = None

# -----------------------------
# 6. Build GenAI Graph (LAZY INIT)
# -----------------------------
def build_genai_graph():
    logging.info("Initializing GenAI RAG pipeline...")

    from dotenv import load_dotenv
    from google import genai

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import CharacterTextSplitter

    from langgraph.graph import StateGraph, START, END

    load_dotenv()

    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
        http_options={"api_version": "v1"}
    )

    # -----------------------------
    # RAG setup
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


    file_path = "data/docs.txt"

    if os.path.exists(file_path):
        loader = TextLoader(file_path)
        documents = loader.load()
    else:
        documents = ["LangGraph is a framework for stateful LLM workflows."]

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    from langchain_core.documents import Document
    docs = splitter.split_documents(
        [d if isinstance(d, Document) else Document(page_content=str(d)) for d in documents]
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    CONFIDENCE_THRESHOLD = 1600

    # -----------------------------
    # LangGraph State
    # -----------------------------
    class AgentState(TypedDict):
        question: str
        context: str
        answer: str
        best_distance: float
        gap: float



    # -----------------------------
    # Graph Nodes
    # -----------------------------
    def retrieve_node(state: AgentState):
        results = vectorstore.similarity_search_with_score(
            state["question"], k=3
        )

        if not results:
            return {"context": "", "best_distance": 999, "gap": 0}

        docs, scores = zip(*results)
        sorted_scores = sorted(scores)

        best = sorted_scores[0]
        second = sorted_scores[1] if len(sorted_scores) > 1 else 999
        gap = second - best

        context = "\n".join(d.page_content for d in docs)

        logging.info(f"Best distance: {best:.2f}, gap: {gap:.2f}")

        return {
            "context": context,
            "best_distance": best,
            "gap": gap
        }

    def decide_node(state: AgentState):
        return {}


    RAG_THRESHOLD = 1.2
    FALLBACK_THRESHOLD = 2.5
    GAP_THRESHOLD = 0.25

    def route_decision(state: AgentState):
        d = state["best_distance"]
        gap = state["gap"]

        # Garbage / no retrieval → block hallucination
        if d > FALLBACK_THRESHOLD:
            logging.info("Very weak retrieval → FALLBACK")
            return "fallback"

        # Strong document grounding
        if d < RAG_THRESHOLD and gap > GAP_THRESHOLD:
            logging.info("Strong retrieval → RAG")
            return "rag"

        # Otherwise allow general LLM knowledge
        logging.info("Uncertain → LLM_ONLY")
        return "llm_only"


    def rag_answer_node(state: AgentState):
        prompt = f"""
Use ONLY the context below to answer.
If context is insufficient, say you don't know.

Context:
{state['context']}

Question:
{state['question']}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return {"answer": response.text}

    def llm_only_node(state: AgentState):
        prompt = f"""
Answer using general knowledge.

Question:
{state['question']}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return {"answer": response.text}

    def fallback_node(state: AgentState):
        return {
            "answer": "I don’t have enough reliable information to answer this question."
        }

    # -----------------------------
    # Build LangGraph
    # -----------------------------
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("rag_answer", rag_answer_node)
    workflow.add_node("llm_answer", llm_only_node)
    workflow.add_node("fallback", fallback_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "decide")

    workflow.add_conditional_edges(
        "decide",
        route_decision,
        {
            "rag": "rag_answer",
            "llm_only": "llm_answer",
            "fallback": "fallback"
        }
    )

    workflow.add_edge("rag_answer", END)
    workflow.add_edge("llm_answer", END)
    workflow.add_edge("fallback", END)

    logging.info("GenAI RAG pipeline initialized successfully")

    return workflow.compile()

# -----------------------------
# 7. Graph accessor (SAFE)
# -----------------------------
def get_graph():
    global graph_app
    if graph_app is None:
        graph_app = build_genai_graph()
    return graph_app

# -----------------------------
# 8. FastAPI Endpoints
# -----------------------------
@api.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()
    logging.info(f"Question received: {req.question}")

    graph = get_graph()
    result = graph.invoke({"question": req.question})

    logging.info(f"Response time: {time.time() - start:.2f}s")
    return {"answer": result["answer"]}

@api.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# 9. CLI test (optional)
# -----------------------------
if __name__ == "__main__":
    g = get_graph()
    q = input("Ask a question: ")
    r = g.invoke({"question": q})
    print(r["answer"])
