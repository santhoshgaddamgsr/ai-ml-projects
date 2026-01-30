# 🧠 Deterministic Agentic AI — RAG with LangGraph, Gemini & Docker

This project demonstrates a **production-oriented deterministic agentic AI system** that answers user questions using:

- **Company knowledge (RAG)**
- **Large Language Model (Google Gemini)**
- **Rule-based routing using semantic similarity**

The agent **deterministically decides** whether to use internal company documents or fall back to the LLM based on **vector similarity thresholds**, ensuring predictable and enterprise-safe behavior.

---

## 🚀 Architecture
```
User Question
↓
Vector Similarity Search (FAISS)
↓
Is the question related to company knowledge?
↓
┌──────────────────────────┐
│ YES → RAG Tool │
│ • Company documents │
│ • Semantic retrieval │
│ • Grounded response │
└──────────────────────────┘
↓
┌──────────────────────────┐
│ NO → LLM Tool │
│ • Gemini model │
│ • General knowledge │
└──────────────────────────┘
↓
Final Answer
```

This design minimizes hallucinations and provides **controlled AI behavior suitable for enterprise environments**.

---

## 📚 Company Knowledge Base

The following internal knowledge is used for retrieval:

- Employees receive **20 days of paid leave per year**
- **Maternity leave duration is 6 months**
- **Work-from-home is allowed 2 days per week**

These documents are stored in a **FAISS vector database** and retrieved using **semantic search**.

---

## 🧠 Agent Type

This system implements a **deterministic agentic pipeline**.

- Routing decisions are **rule-based**
- LLM is **not allowed to control execution flow**
- Same input always produces the same routing decision

This design prioritizes:

- predictability
- auditability
- cost control
- reduced hallucination risk

---

## ✨ Key Features

- LangGraph-based agent orchestration  
- Deterministic routing using similarity thresholds  
- FAISS vector store for document retrieval  
- HuggingFace MiniLM embeddings  
- Google Gemini for response generation  
- FastAPI backend service  
- Fully Dockerized deployment  

---

## 🛠️ How to Run (Docker)

### 1️⃣ Create `.env`

Create a file named `.env`:
```
GOOGLE_API_KEY=your_gemini_api_key
```

### 2️⃣ Build Docker image
```
docker build -t deterministic-agent .
```
### 3️⃣ Run the application
```
docker run -p 8000:8000 deterministic-agent
```
### 4️⃣ Test in browser
Open:
```
http://localhost:8000/docs
```
Use POST /ask with:
```
{
  "question": "What is maternity leave?"
}
```
### 🎯 What This Project Demonstrates
This project reflects real-world GenAI engineering practices:  
- Deterministic agentic routing  
- Retrieval-Augmented Generation (RAG)  
- Enterprise-safe AI architecture  
- Similarity-based decision control  
- API-based AI deployment  
- Containerized production setup  

This is the type of architecture commonly used in enterprise internal assistants, where reliability and governance are more important than autonomous reasoning.

### 👨‍💻 Author
Santhosh Gaddam
