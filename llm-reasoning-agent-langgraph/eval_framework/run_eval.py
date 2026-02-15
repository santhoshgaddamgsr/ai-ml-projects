import os
import sys
import time
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from deepeval import assert_test

from google import genai
from google.genai import errors

load_dotenv()

# ======================================================
# Paths
# ======================================================
CURRENT_DIR = Path(__file__).parent.resolve()
DATASET_PATH = CURRENT_DIR / "evaluation_dataset.csv"
RESULTS_PATH = CURRENT_DIR / "evaluation_results_detailed.csv"
HISTORY_PATH = CURRENT_DIR / "evaluation_history.csv"

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

# ======================================================
# Import RAG pipeline
# ======================================================
PROJECT_ROOT = str(CURRENT_DIR.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import rag_pipeline


# ======================================================
# Gemini Judge
# ======================================================
class VertexGeminiJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location="us-central1"
        )

    def load_model(self):
        return self.client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=(lambda e: isinstance(e, errors.ClientError) and "429" in str(e))
    )
    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# ======================================================
# Run Evaluation Sequentially
# ======================================================
if __name__ == "__main__":
    print("🚀 Running RAG Evaluation (Sequential Mode)...\n")

    judge = VertexGeminiJudge()
    df = pd.read_csv(DATASET_PATH)

    metrics = [
        FaithfulnessMetric(model=judge, async_mode=False),
        AnswerRelevancyMetric(model=judge, async_mode=False),
        ContextualPrecisionMetric(model=judge, async_mode=False),
    ]

    all_results = []

    for idx, row in df.iterrows():
        print(f"\n🔍 Evaluating Test Case {idx + 1}")

        result = rag_pipeline(row["question"])

        test_case = LLMTestCase(
            input=row["question"],
            actual_output=result.get("answer", ""),
            retrieval_context=result.get("contexts", []),
            expected_output=str(row.get("ground_truth", ""))
        )

        assert_test(test_case, metrics, run_async=False)

        faithfulness = metrics[0].score
        relevancy = metrics[1].score
        precision = metrics[2].score

        print("Faithfulness:", faithfulness)
        print("Answer Relevancy:", relevancy)
        print("Context Precision:", precision)

        all_results.append({
            "question": row["question"],
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision
        })

        # ⏳ Delay only if NOT last test case
        if idx < len(df) - 1:
            print("⏳ Waiting 60 seconds to avoid Vertex rate limits...")
            time.sleep(60)

    # ======================================================
    # Save Detailed Per-Test Results
    # ======================================================
    detailed_df = pd.DataFrame(all_results)
    detailed_df.to_csv(RESULTS_PATH, index=False)

    print("\n📁 Detailed results saved to:", RESULTS_PATH)

    # ======================================================
    # Calculate Averages
    # ======================================================
    avg_faithfulness = detailed_df["faithfulness"].mean()
    avg_relevancy = detailed_df["answer_relevancy"].mean()
    avg_precision = detailed_df["context_precision"].mean()

    print("\n📊 Average Metrics:")
    print("Average Faithfulness:", round(avg_faithfulness, 3))
    print("Average Answer Relevancy:", round(avg_relevancy, 3))
    print("Average Context Precision:", round(avg_precision, 3))

    # ======================================================
    # Threshold Validation
    # ======================================================
    THRESHOLD = 0.7

    if (
        avg_faithfulness >= THRESHOLD and
        avg_relevancy >= THRESHOLD and
        avg_precision >= THRESHOLD
    ):
        status = "PASS ✅"
    else:
        status = "FAIL ❌"

    print("\n🎯 Evaluation Status:", status)

    # ======================================================
    # Save Summary History
    # ======================================================
    history_exists = HISTORY_PATH.exists()

    with open(HISTORY_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not history_exists:
            writer.writerow([
                "timestamp",
                "avg_faithfulness",
                "avg_relevancy",
                "avg_precision",
                "status"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(avg_faithfulness, 3),
            round(avg_relevancy, 3),
            round(avg_precision, 3),
            status
        ])

    print("📁 Evaluation history updated:", HISTORY_PATH)
    print("\n✅ Evaluation Completed Successfully")
