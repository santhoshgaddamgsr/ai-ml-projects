import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# LangGraph
from langgraph.graph import StateGraph, START, END

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


# ======================================================
# 1. Load environment
# ======================================================
load_dotenv()


# ======================================================
# 2. LLM
# ======================================================
USE_VERTEX = os.getenv("USE_VERTEX", "false").lower() == "true"

if USE_VERTEX:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=os.getenv("GCP_PROJECT_ID"),
        location=os.getenv("LOCATION", "asia-south1"),
    )
    print("✅ Using Vertex AI Gemini")
else:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    print("✅ Using Google AI Studio API Key")


# ======================================================
# 3. Load RAG documents
# ======================================================
DOC_PATH = "docs/sample.txt"

loader = TextLoader(DOC_PATH, encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)


retriever = vectorstore.as_retriever()

# ======================================================
# Retrieval confidence (FAISS → 0–1 trust score)
# ======================================================
def compute_rag_confidence(query: str):
    results = vectorstore.similarity_search_with_score(query, k=1)

    if not results:
        return None

    _, distance = results[0]
    distance = float(distance)

    # 🔧 calibrated for your docs
    MIN_DISTANCE = 0.7
    MAX_DISTANCE = 1.6

    if distance <= MIN_DISTANCE:
        return 1.0

    if distance >= MAX_DISTANCE:
        return 0.0

    # linear normalization
    confidence = 1 - ((distance - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE))

    return round(confidence, 2)


# ======================================================
# 4. Agent State
# ======================================================
class AgentState(TypedDict):
    user_input: str
    route: str
    reason: str
    final_answer: str
    confidence: float | None


# ======================================================
# 5. Routing schema
# ======================================================
class RoutingDecision(BaseModel):
    action: Literal["rag", "llm"]
    reason: str


# ======================================================
# 6. Similarity scoring
# ======================================================
def get_similarity_score(query: str):
    results = vectorstore.similarity_search_with_score(query, k=1)
    if not results:
        return None
    _, score = results[0]
    return score


# ======================================================
# 7. Reasoning Node
# ======================================================
def reasoning_node(state: AgentState) -> AgentState:

    score = get_similarity_score(state["user_input"])

    # ---- RAG decision ----
    if score is not None and score < 0.6:
        return {
            "user_input": state["user_input"],
            "route": "rag",
            "reason": "Relevant internal documents detected using vector similarity",
            "confidence": round(1 - score, 2),
            "final_answer": ""
        }

    # ---- LLM decision ----
    parser = PydanticOutputParser(pydantic_object=RoutingDecision)

    prompt = PromptTemplate(
        template="""
You are an enterprise AI reasoning agent.

Decide response source.

Rules:
- rag → internal company documentation required
- llm → general knowledge or reasoning

{format_instructions}

Question:
{question}
""",
        input_variables=["question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    decision: RoutingDecision = (prompt | llm | parser).invoke(
        {"question": state["user_input"]}
    )

    return {
        "user_input": state["user_input"],
        "route": decision.action,
        "reason": decision.reason,
        "confidence": None,
        "final_answer": ""
    }


# ======================================================
# 8. Action Nodes
# ======================================================
def rag_node(state: AgentState) -> AgentState:
    docs = retriever.invoke(state["user_input"])
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY using the context below.

If not found, say:
"I don't know based on internal documents."

Context:
{context}

Question:
{state["user_input"]}
"""

    answer = llm.invoke(prompt).content

    return {
        "user_input": state["user_input"],
        "route": state["route"],
        "reason": state["reason"],
        "confidence": state["confidence"],  # ✅ PRESERVE
        "final_answer": answer
    }



def llm_node(state: AgentState) -> AgentState:
    answer = llm.invoke(state["user_input"]).content

    return {
        "user_input": state["user_input"],
        "route": state["route"],
        "reason": state["reason"],
        "confidence": None,
        "final_answer": answer
    }



# ======================================================
# 9. Router
# ======================================================
def router(state: AgentState) -> str:
    return state["route"]


# ======================================================
# 10. LangGraph
# ======================================================
graph = StateGraph(AgentState)

graph.add_node("reasoning", reasoning_node)
graph.add_node("rag", rag_node)
graph.add_node("llm", llm_node)

graph.add_edge(START, "reasoning")

graph.add_conditional_edges(
    "reasoning",
    router,
    {
        "rag": "rag",
        "llm": "llm",
    }
)

graph.add_edge("rag", END)
graph.add_edge("llm", END)

agent_app = graph.compile()


# ======================================================
# 11. FastAPI + UI
# ======================================================
class QueryRequest(BaseModel):
    query: str


app = FastAPI(title="Enterprise Agentic AI Assistant")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Enterprise Agentic AI Assistant</title>
<style>
body { font-family: Arial; background:#f6f8fa; }
.container { max-width:900px; margin:40px auto; background:white;
padding:30px; border-radius:10px; }
.badge { padding:4px 12px; border-radius:20px; font-size:13px; }
.rag { background:#dcfce7; color:#166534; }
.llm { background:#e0f2fe; color:#075985; }
.warn { color:#92400e; font-size:13px; }
.example button { margin:5px; }
</style>
</head>

<body>
<div class="container">
<h2>Enterprise Agentic AI Assistant</h2>
<p>LLM reasoning • Retrieval grounding • Explainability</p>

<div class="example">
<b>Try examples:</b><br>
<button onclick="setQ('What is LangGraph?')">LangGraph</button>
<button onclick="setQ('What is the work from home policy?')">WFH Policy</button>
<button onclick="setQ('What is leave policy?')">Leave Policy</button>
<button onclick="setQ('What is maternity policy?')">Maternity Policy</button>
</div>

<br>
<textarea id="q" style="width:100%;height:70px;"></textarea><br><br>
<button onclick="ask()">Ask</button>

<div id="out" style="display:none;margin-top:20px;">
<hr>

<b>Answer:</b>
<div id="answer"></div>

<br><br>
<b>Route Used:</b>
<span id="route" class="badge"></span>

<br><br>
<div id="confidence"></div>
<div id="reason"></div>
</div>

<script>
function setQ(q){ document.getElementById("q").value=q; }

async function ask(){
 const q=document.getElementById("q").value;
 const r=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:q})});
 const d=await r.json();

 document.getElementById("answer").innerText=d.answer;

 const route=document.getElementById("route");
 route.innerText=d.route.toUpperCase();
 route.className="badge "+d.route;

 const conf=document.getElementById("confidence");

 if(d.route==="rag"){
   let label =
  d.confidence > 0.6 ? "🟢 High" :
  d.confidence > 0.3 ? "🟡 Medium" :
  "🔴 Low";

conf.innerHTML =
  "<b>Retrieval Confidence:</b> " + d.confidence + " (" + label + ")";

 } else {
   conf.innerHTML="<span class='warn'>⚠ Generated by AI model (not grounded in internal documents)</span>";
 }

 document.getElementById("reason").innerHTML="<br><b>Reason:</b> "+d.reason;
 document.getElementById("out").style.display="block";
}
</script>
</body>
</html>
"""


@app.post("/ask")
def ask(req: QueryRequest):

    # 1️⃣ compute retrieval confidence BEFORE agent runs
    confidence = compute_rag_confidence(req.query)

    # 2️⃣ run LangGraph agent
    result = agent_app.invoke({
        "user_input": req.query
    })

    # 3️⃣ return confidence ONLY for RAG
    return {
        "route": result["route"],
        "reason": result["reason"],
        "answer": result["final_answer"],
        "confidence": confidence if result["route"] == "rag" else None,
    }


