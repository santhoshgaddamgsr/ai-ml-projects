from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from typing import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel


# -------------------------
# 1. LLM
# -------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# -------------------------
# 2. Company documents
# -------------------------
docs = [
    Document(page_content="Employees get 20 days of paid leave per year."),
    Document(page_content="Maternity leave is 6 months."),
    Document(page_content="Work from home is allowed 2 days per week.")
]

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# -------------------------
# 3. Similarity routing
# -------------------------
ROUTING_THRESHOLD = 0.08

def get_similarity(query):
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=1)
    distance = docs_with_scores[0][1]
    return 1 - distance

# -------------------------
# 4. Memory (topic context)
# -------------------------
conversation_memory = {
    "last_topic": None
}

# -------------------------
# 5. State
# -------------------------
class AgentState(TypedDict):
    question: str
    answer: str
class QueryRequest(BaseModel):
    question: str


# -------------------------
# 6. Tools
# -------------------------
def rag_tool(state):
    query = state["question"]

    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    conversation_memory["last_topic"] = context

    prompt = f"""
You are answering using company documents.

Context:
{context}

Question:
{query}

Answer clearly from the documents.
"""
    response = llm.invoke(prompt)
    return {"answer": response.content}

def llm_tool(state):
    query = state["question"]

    if conversation_memory["last_topic"]:
        prompt = f"""
Previous conversation context:
{conversation_memory["last_topic"]}

User question:
{query}

If relevant, use previous context.
"""
    else:
        prompt = query

    response = llm.invoke(prompt)
    return {"answer": response.content}

# -------------------------
# 7. Agent Brain (Routing)
# -------------------------
def agent_decide(state):
    query = state["question"]
    similarity = get_similarity(query)

    print(f"\n[Routing similarity score]: {similarity}")

    if similarity > ROUTING_THRESHOLD:
        return "rag"
    else:
        return "llm"

# -------------------------
# 8. LangGraph
# -------------------------
graph = StateGraph(AgentState)

graph.add_node("rag", rag_tool)
graph.add_node("llm", llm_tool)
graph.add_node("agent", lambda state: state)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    agent_decide,
    {
        "rag": "rag",
        "llm": "llm"
    }
)

graph.set_finish_point("rag")
graph.set_finish_point("llm")

agent_app = graph.compile()
fastapi_app = FastAPI()
@fastapi_app.post("/ask")
def ask_agent(request: QueryRequest):
    result = agent_app.invoke({"question": request.question})
    return {
        "question": request.question,
        "answer": result["answer"]
    }



