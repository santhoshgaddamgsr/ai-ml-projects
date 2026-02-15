import os
from typing import TypedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse

# LangChain / LangGraph
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph

# Globals
llm = None
vectorstore = None
graph_app = None

FAISS_PATH = "faiss_index"


# ======================================================
# 1. Lifespan (startup loading) – OPTIMIZED ONLY
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, vectorstore, graph_app

    load_dotenv()

    # -----------------------------
    # LLM setup (UNCHANGED LOGIC)
    # -----------------------------
    from langchain_google_genai import ChatGoogleGenerativeAI

    if os.getenv("USE_VERTEX", "false").lower() == "true":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("LOCATION", "asia-south1"),
        )
        print("✅ Using Vertex AI Gemini")
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        print("✅ Using Google AI Studio API Key")

    # -----------------------------
    # Vectorstore setup – OPTIMIZED
    # -----------------------------
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 🔥 FAST PATH – Load existing index
    if os.path.exists(FAISS_PATH):
        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("⚡ Loaded existing FAISS index")

    else:
        print("📚 Building FAISS index first time...")

        loader = DirectoryLoader(
            "data/knowledge_base",
            glob="*.txt",
            loader_cls=TextLoader
        )

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )

        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)

        # 🔥 SAVE FOR NEXT STARTUP
        vectorstore.save_local(FAISS_PATH)
        print("💾 FAISS index saved")

    # -----------------------------
    # Compile graph
    # -----------------------------
    graph_app = compile_graph()

    # 🔥 Warmup (no logic change)
    try:
        vectorstore.similarity_search("warmup", k=1)
        llm.invoke("hi")
    except:
        pass

    print("🚀 Startup complete")
    yield


# ======================================================
# 2. Agent State (UNCHANGED)
# ======================================================
class AgentState(TypedDict):
    question: str
    answer: str
    route: str
    confidence: float
    reason: str


# ======================================================
# 3. Similarity-based routing (EXACT SAME LOGIC)
# ======================================================
def decide_route_with_confidence(query: str):
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=1)
    distance = docs_with_scores[0][1]

    THRESHOLD = 1.2

    if distance < THRESHOLD:
        confidence = round(1 - (distance / THRESHOLD), 2)
        return "rag", confidence
    else:
        return "llm", 0.0


# ======================================================
# 4. Tools (UNCHANGED)
# ======================================================
def rag_tool(state: AgentState):
    query = state["question"]

    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an HR assistant.

Context:
{context}

Question:
{query}

Answer strictly from context.
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response.content if hasattr(response, "content") else response,
    }


def llm_tool(state: AgentState):
    response = llm.invoke(state["question"])

    return {
        **state,
        "answer": response.content if hasattr(response, "content") else response,
    }


def agent_node(state: AgentState):
    route, confidence = decide_route_with_confidence(state["question"])

    return {
        **state,
        "route": route,
        "confidence": round(confidence, 2),
        "reason": (
            "Answer retrieved from internal HR documents"
            if route == "rag"
            else "Question outside enterprise knowledge base"
        ),
    }


# ======================================================
# 5. LangGraph (UNCHANGED)
# ======================================================
def compile_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("rag", rag_tool)
    workflow.add_node("llm", llm_tool)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        lambda s: s["route"],
        {
            "rag": "rag",
            "llm": "llm",
        },
    )

    workflow.set_finish_point("rag")
    workflow.set_finish_point("llm")

    return workflow.compile()


# ======================================================
# 6. FastAPI + HTML (100% UNTOUCHED)
# ======================================================
app = FastAPI(lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    route: str
    confidence: float
    reason: str


@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Enterprise Agentic AI Assistant</title>
</head>

<body style="font-family: Arial; margin: 40px;">

<h1>Enterprise Agentic AI Assistant</h1>

<p style="color: gray;">
Agentic loop-based GenAI system with controlled decision routing, retrieval grounding, and explainability.
</p>

<hr>

<b>Try example questions:</b><br><br>

<button onclick="setQ('How many leave days do employees get?')">Leave policy</button>
<button onclick="setQ('What is the maternity leave duration?')">Maternity policy</button>
<button onclick="setQ('Is work from home allowed?')">WFH policy</button>
<button onclick="setQ('Who is MS Dhoni?')">Out of scope</button>

<br><br>

<input id="question"
       style="width: 70%; padding: 10px;"
       placeholder="Ask a question..." />

<button onclick="ask()">Ask</button>

<pre id="result" style="margin-top:20px; padding:10px; background:#f7f7f7;"></pre>

<script>
function setQ(q) {
    document.getElementById("question").value = q;
}

async function ask() {
    const q = document.getElementById("question").value;

    const res = await fetch("/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: q})
    });

    const data = await res.json();

    let confidenceText = data.confidence > 0.5 ? "🟢 High" :
                         data.confidence > 0.2 ? "🟡 Medium" :
                         "🔴 Low";

    document.getElementById("result").innerText =
        "Answer:\\n" + data.answer + "\\n\\n" +
        "Route Used: " + data.route.toUpperCase() + "\\n" +
        "Retrieval Confidence: " + data.confidence + " (" + confidenceText + ")\\n" +
        "Reason: " + data.reason;
}
</script>

</body>
</html>"""


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    result = graph_app.invoke({"question": req.question})
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=True,
    )
