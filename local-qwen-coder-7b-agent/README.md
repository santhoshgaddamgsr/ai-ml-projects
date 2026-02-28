# 🚀 Local Qwen Coding Agent (7B)

A fully local AI-powered coding assistant built using **Qwen2.5-Coder 7B**, **Ollama**, **FastAPI**, and a browser-based UI.

This project demonstrates how to run and integrate open-source LLMs locally to perform:

* Code generation
* Bug fixing
* Code review

without any external APIs or cloud dependencies.

---

## 🧠 Features

✅ Local LLM inference (no API cost)
✅ GPU acceleration support (via Ollama)
✅ FastAPI backend
✅ Simple browser UI
✅ Multiple agent modes:

* Generate
* Fix
* Review
  ✅ Modular architecture for future upgrades

---

## 🏗️ Architecture

```
Browser UI
    ↓
FastAPI Backend
    ↓
Ollama Runtime
    ↓
Qwen2.5-Coder 7B
    ↓
GPU / CPU
```

---

## 📂 Project Structure

```
local-qwen-coder-7b-agent/
│
├── backend/
│   ├── main.py
│   └── venv/
│
├── frontend/
│   └── index.html
│
└── README.md
```

---

## ⚙️ Requirements

* Python 3.10+
* Ollama installed
* Qwen model downloaded
* Optional GPU (recommended)

---

## 🤖 Install Ollama

Linux / WSL:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:

```bash
ollama --version
```

---

## 📥 Download Model

```bash
ollama pull qwen2.5-coder:7b
```

---

## 🐍 Setup Backend

Navigate to backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install fastapi uvicorn requests pydantic python-dotenv
```

---

## ▶️ Run Backend

```bash
uvicorn main:app --reload --port 8002
```

API docs:

```
http://127.0.0.1:8002/docs
```

---

## 🌐 Run Frontend

Open:

```
frontend/index.html
```

in your browser.

Make sure backend is running.

---

## 🧪 Example Prompts

### Generate Code

```
Create a FastAPI CRUD API for users with SQLite
```

### Fix Code

```
def add(a,b):
    return a-b
```

### Review Code

```
password = "admin123"
if input() == password:
    print("login success")
```

---

## 🚀 Future Improvements

* File editing agent
* Repository understanding (RAG)
* Multi-model support
* Syntax highlighting
* Chat history
* Docker deployment

---

## 🎯 Learning Outcomes

This project demonstrates:

* Local LLM deployment
* AI agent design
* Backend integration
* Prompt engineering
* Full-stack AI application development

---

## 👨‍💻 Author

**Santhosh Gaddam**

AI/ML Engineer | GenAI Systems | LLM Applications

GitHub: https://github.com/santhoshgaddamgsr

---

## ⭐ Acknowledgements

* Alibaba Qwen Models
* Ollama Runtime
* FastAPI
