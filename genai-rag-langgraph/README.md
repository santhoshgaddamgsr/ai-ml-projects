
# 🧠 GenAI RAG System – LangGraph + FastAPI

A production-style Retrieval-Augmented Generation (RAG) system built using LangGraph, FAISS, HuggingFace embeddings, Gemini LLM, and FastAPI.

This project demonstrates how to build a hallucination-resistant GenAI backend that intelligently decides when to use retrieved documents vs general LLM knowledge.

---

## 🚀 What this project does

This system accepts a user question and dynamically decides:

- Use retrieved documents (RAG) when relevant context is available  
- Use the LLM directly when documents are weak  
- Reject answering when confidence is too low  

This is done using:
- Vector distance
- Score gap between best & second match
- Threshold-based routing

This is how real enterprise GenAI systems avoid hallucinations.

---

## 🧩 Architecture

User  
→ FastAPI (/ask)  
→ LangGraph State Machine  
→ FAISS Vector Search (HuggingFace Embeddings)  
→ Decision Router (distance + gap)  
→ RAG / LLM / Fallback  

---

## 🧠 Key Features

- LangGraph-based stateful workflow  
- FAISS semantic search  
- HuggingFace embeddings (all-MiniLM-L6-v2)  
- Gemini 2.5 Flash for generation  
- Distance + gap-based hallucination control  
- FastAPI production API  
- Fully reproducible with requirements.txt  

---

## 📁 Project Structure

genai-rag-langgraph/
│
├── main.py
├── requirements.txt
├── .env.example
├── data/
│ └── docs.txt
└── .gitignore



---

## 🛠️ Setup & Run

### 1. Clone the repo
git clone <your-github-repo-url>
cd genai-rag-langgraph



### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate



### 3. Install dependencies
pip install -r requirements.txt



### 4. Set Gemini API key

Create a file `.env`:
GOOGLE_API_KEY=your_gemini_api_key_here



### 5. Run the API
uvicorn main:api --reload



Open:
http://127.0.0.1:8000/docs



---

## 🧪 Example

POST `/ask`
{
"question": "What is LangGraph?"
}



The system will retrieve documents, evaluate confidence, and generate a grounded response.

---

## 🎯 Why this matters

This project demonstrates:

- RAG engineering
- Vector search
- LLM integration
- API deployment
- Hallucination control
- Production-style GenAI design



---

## 👨‍💻 Author

Santhosh Gaddam