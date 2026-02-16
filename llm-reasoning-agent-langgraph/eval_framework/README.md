# 📊 RAG Evaluation Framework (DeepEval)

This folder contains the **evaluation framework** for the LangGraph-based LLM Reasoning Agent.

The framework uses **DeepEval + Gemini (Vertex AI)** as a judge model to evaluate the quality and grounding of the RAG system.

---

## 🎯 Purpose

This framework measures:

- ✅ Faithfulness (Is the answer grounded in retrieved context?)
- ✅ Answer Relevancy (Is the answer relevant to the question?)
- ✅ Context Precision (Are retrieved documents relevant?)
- 📊 Average metric scoring
- 🎯 Threshold-based PASS / FAIL validation

This ensures the RAG system meets enterprise-grade reliability standards.

---

## 📁 Files

```
eval_framework/
│
├── run_eval.py
├── evaluation_dataset.csv
├── evaluation_results_detailed.csv (generated)
├── evaluation_history.csv (generated)
└── README.md
```

---

## 🧠 Metrics Used

### 1️⃣ Faithfulness
Checks whether the answer is supported by retrieved context.

### 2️⃣ Answer Relevancy
Measures whether the response directly answers the user’s question.

### 3️⃣ Context Precision
Evaluates whether retrieved documents are actually relevant.

---

## ⚙️ Setup

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Ensure environment variables are set:

```bash
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=path_to_service_account.json
```

---

## ▶️ Run Evaluation

From project root:

```bash
python llm-reasoning-agent-langgraph/eval_framework/run_eval.py
```

---

## 📊 Output

### 1️⃣ Detailed per-test metrics

Saved to:

```
evaluation_results_detailed.csv
```

Contains:
- Question
- Faithfulness score
- Answer Relevancy score
- Context Precision score

---

### 2️⃣ Evaluation History

Saved to:

```
evaluation_history.csv
```

Contains:
- Timestamp
- Average Faithfulness
- Average Answer Relevancy
- Average Context Precision
- PASS / FAIL status

---

## 🎯 Threshold Logic

The evaluation passes only if:

- Faithfulness ≥ 0.8
- Answer Relevancy ≥ 0.8
- Context Precision ≥ 0.8

These thresholds simulate enterprise validation standards.

---

## 🏢 Why This Matters

In production environments:

- Each test case metric is stored
- Average metrics are calculated
- System is validated against defined thresholds
- Evaluation history is maintained for regression tracking

This enables:

- Continuous quality monitoring
- Model improvement tracking
- Release validation

---

## 🚀 Enterprise Practice

This evaluation framework is:

- Separate from runtime application
- Not deployed to production
- Used for quality validation and benchmarking
- Version controlled for reproducibility

---

## 👨‍💻 Author

Santhosh Gaddam  
LLM Systems | RAG | Agentic Architectures
