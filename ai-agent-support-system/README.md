# 🧠 Agentic AI – RAG with LangGraph, Gemini & Docker

This project is a **production-grade Agentic AI system** that answers questions using:
- Company knowledge (RAG)
- Large Language Model (Gemini)
- Intelligent routing using semantic similarity

The agent decides **dynamically** whether to use company documents or the LLM based on how relevant the question is.

---

## 🚀 Architecture

```

User Question
↓
Vector Similarity (FAISS)
↓
Is it related to company data?
↓
YES → RAG Tool (Docs + Embeddings)
NO → LLM Tool (Gemini)
↓
Final Answer

```

This avoids hallucinations and makes the system safe for enterprise use.


---

## 📚 Company Knowledge Used

- Employees get 20 days of paid leave per year  
- Maternity leave is 6 months  
- Work from home is allowed 2 days per week  

These are stored in a vector database and retrieved using semantic search.

---

## 🧠 Key Features

- LangGraph based agent routing  
- FAISS vector store for document retrieval  
- HuggingFace embeddings (MiniLM)  
- Gemini LLM for reasoning  
- Conversation memory  
- FastAPI service  
- Fully Dockerized  

---

## 🛠️ How to Run (Docker)

### 1️⃣ Create `.env`

Create a file named `.env`:

GOOGLE_API_KEY=your_gemini_api_key


---

### 2️⃣ Build the Docker image

```bash
docker build -t agentic-ai .
```
---
3️⃣ Run the AI

docker run -p 8000:8000 agentic-ai

---
4️⃣ Test in browser

Open:
http://localhost:8000/docs
Use POST /ask with:
{
  "question": "What is maternity leave?"
}

---
🧑‍💻 What this demonstrates

This project demonstrates real-world GenAI engineering:

    Agentic decision making
    Retrieval-Augmented Generation (RAG)
    Safe AI routing using similarity thresholds
    API-based deployment
    Containerized production setup
This is the same architecture used in enterprise AI assistants.

📌 Author
Built by Santhosh Gaddam as part of an AI / GenAI engineering portfolio.
