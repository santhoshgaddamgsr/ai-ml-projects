# 🤖 AI LLM Reasoning Agent using LangGraph

An enterprise-style **Agentic AI system** built using **LangGraph + LangChain + Gemini LLM**, capable of **intelligent decision-making between RAG and direct LLM reasoning**, with **explainability and retrieval confidence scoring**.

This project demonstrates how modern GenAI systems go beyond simple chatbots by introducing **reasoning-based routing**, **grounded responses**, and **transparent decision logic**.

---

## 🚀 Key Features

✅ LLM-based reasoning agent (LangGraph)  
✅ Hybrid routing: **RAG vs Direct LLM**  
✅ Vector similarity–based confidence scoring  
✅ Explainable AI decisions (`reason` field)  
✅ FAISS vector database  
✅ HuggingFace sentence-transformer embeddings  
✅ FastAPI backend with interactive UI  
✅ Google Gemini support (AI Studio & Vertex AI)  
✅ Dockerized and deployed on Google Cloud Run  

---

## 🧠 What makes this Agentic AI?

Unlike traditional chatbots or basic RAG pipelines, this system:

- **Reasons before answering**
- **Decides the source of truth**
- **Explains why a route was chosen**
- **Shows confidence when using internal documents**

### Decision flow:
```
User Question
↓
Reasoning Agent (LangGraph)
↓
Evaluate relevance of internal knowledge
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

## 🧩 Architecture
```
FastAPI UI
│
▼
LangGraph Reasoning Node
│
├── Vector similarity check
│
├── LLM-based routing decision
│
▼
Conditional Execution
├── RAG Node (FAISS + Docs)
└── LLM Node (Gemini)
```

---

## 🔀 Routing Logic

### 1️⃣ Vector-based decision
- Uses FAISS similarity score
- High similarity → internal documentation likely relevant

### 2️⃣ LLM-based reasoning
If similarity is unclear, Gemini LLM decides:

```json
{
  "action": "rag | llm",
  "reason": "why this route is chosen"
}
```
This makes the system interpretable and explainable.
---

## 📊 Retrieval Confidence Scoring
When RAG is used:  
- FAISS distance is normalized into 0–1 confidence  
- Displayed as:  
    🟢 High confidence  
    🟡 Medium confidence  
    🔴 Low confidence
  
This helps users understand trust level of the answer.  

## 🖥️ UI Preview
The FastAPI UI shows:
- Answer
- Route used (RAG / LLM)
- Retrieval confidence
- Reasoning explanation

This is ideal for enterprise demos and PoCs.

---
## 📁 Project Structure
```
ai-llm-reasoning-agent-langgraph/
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
## ⚙️ Tech Stack
- Python 3.11
- FastAPI
- LangGraph
- LangChain
- Google Gemini (2.5 Flash / Flash Lite)
- FAISS
- HuggingFace Embeddings
- Docker
- Google Cloud Run

## ▶️ Run Locally
### 1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Create .env
```
GOOGLE_API_KEY=your_api_key
USE_VERTEX=false
```
(For Vertex AI)
```
USE_VERTEX=true
GCP_PROJECT_ID=your_project_id
LOCATION=asia-south1
```
### 4️⃣ Run application
uvicorn main:app --reload
Open:
http://localhost:8000

### 🐳 Run with Docker
docker build -t ai-llm-agent .
docker run -p 8080:8080 ai-llm-agent

## ☁️ Cloud Deployment Overview (Google Cloud Run)
This application has been deployed using **Google Cloud Run** with the following setup:

- Docker-based containerization
- Stateless FastAPI service
- Gemini integration via:
  - Google AI Studio API key
  - Vertex AI (production-ready option)
- Environment-based configuration using `.env`

The same codebase supports both **local development** and **cloud deployment** without modification.

> Note: Public deployment URLs are intentionally not included to avoid dependency on runtime availability and cloud billing.

### 🎯 Use Cases
- Enterprise knowledge assistants
- Internal policy Q&A systems
- Agentic GenAI PoCs
- Explainable RAG systems
- AI architecture demonstrations

### 📌 Key Learning Outcomes
- How to build LLM reasoning agents
- LangGraph conditional routing
- Hybrid RAG architectures
- Confidence calibration
- Explainable GenAI design
- Production-ready GenAI deployment

### 👨‍💻 Author
Santhosh Gaddam
