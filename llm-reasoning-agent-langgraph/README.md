
# 🤖 LLM Reasoning Agent using LangGraph

An enterprise-style **LLM-powered reasoning agent** built using **LangGraph + LangChain + Gemini**, capable of making **intelligent routing decisions between RAG and direct LLM responses**, with full **explainability and confidence scoring**.

This project demonstrates how modern GenAI systems move beyond simple chatbots into **agentic architectures driven by LLM reasoning**.

---

## 🚀 Key Features

✅ LLM-based reasoning for routing decisions  
✅ LangGraph-based agent orchestration  
✅ Hybrid execution: RAG vs direct LLM  
✅ Explainable decision making  
✅ Retrieval confidence scoring  
✅ FAISS vector database  
✅ HuggingFace sentence-transformer embeddings  
✅ FastAPI backend with interactive UI  
✅ Dockerized and Cloud Run compatible  

---

## 🧠 Agent Type

This project implements an **LLM reasoning agent**.

Routing decisions are made **by the LLM itself**, not by fixed rules or thresholds.

The LLM determines:

- whether internal documents are required
- whether general reasoning is sufficient
- why a particular route was chosen

LangGraph is used to **orchestrate and safely execute** the decision made by the LLM.

---

## 🔀 Decision Flow
```
User Question
↓
LLM-based Reasoning & Routing Decision
↓
Agent Orchestration Layer (LangGraph)
↓
┌──────────────────────────┐
│ If internal docs relevant │
│ → RAG Path │
│ • FAISS Retrieval │
│ • Grounded Answer │
└──────────────────────────┘
↓
┌──────────────────────────┐
│ Otherwise │
│ → LLM Path │
│ • Gemini Reasoning │
│ • General Knowledge │
└──────────────────────────┘
↓
Final Answer + Reason + Confidence
```

---

## 📊 Retrieval Confidence

When RAG is selected, vector similarity scores are normalized into a **0–1 confidence score**, providing transparency into how strongly the retrieved documents support the answer.

Confidence levels:

- 🟢 High
- 🟡 Medium
- 🔴 Low

---

## 🧩 Architecture Overview
```
FastAPI UI
↓
LLM Reasoning Node
↓
LangGraph State Machine
↓
Conditional Execution
├── RAG Node (FAISS + Docs)
└── LLM Node (Gemini)
```

---

## 📁 Project Structure
```
llm-reasoning-agent-langgraph/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .gcloudignore
│
└── docs/
└── sample.txt
```


---

## ⚙️ Tech Stack

- Python 3.11
- FastAPI
- LangGraph
- LangChain
- Google Gemini (Flash / Flash Lite)
- FAISS
- HuggingFace Embeddings
- Docker

---

## ▶️ Run Locally

```
pip install -r requirements.txt
uvicorn main:app --reload
```
## 🐳 Docker
```
d111ocker build -t llm-reasoning-agent .
docker run -p 8080:8080 llm-reasoning-agent
```

## ☁️ Deployment Overview
This application has been deployed using Google Cloud Run with Docker-based containerization.  
The same codebase supports both:  
- Google AI Studio API Key
- Vertex AI (production-ready)  
Deployment URLs are intentionally excluded to avoid runtime dependency and billing exposure.

##  🎯 Key Learnings
- Difference between deterministic routing and LLM reasoning  
- Agent orchestration using LangGraph
- Explainable GenAI system design
- Hybrid RAG architectures
- Production-ready GenAI deployment

👨‍💻 Author
Santhosh Gaddam
