import sys, os, json
from pprint import pprint
from termcolor import colored

# ✅ Allow relative imports (modules folder)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.preprocess import TextPreprocessor
from modules.symbolic_retriever import init_symbolic_db, insert_documents
from modules.semantic_retriever import build_semantic_index
from modules.fusion_retriever import hybrid_search

# ──────────────────────────────────────────────────────────────
# 🧠 Step 1: Initialize NLP + RAG systems
# ──────────────────────────────────────────────────────────────

print("=" * 70)
print("🚀 INITIALIZING NLP + HYBRID RAG PIPELINE")
print("=" * 70)

tp = TextPreprocessor()

docs = [
    ("MedicalGPT Ethics", "MedicalGPT, a system by Google DeepMind, advanced medical NLP after 2024."),
    ("Finance AI", "FinAI predicts S&P 500 market trends using transformer models since 2025."),
    ("HybridLM Overview", "HybridLM is a hybrid model combining symbolic and semantic reasoning built by OpenAI in 2025."),
    ("LLaMA Evolution", "LLaMA 3 outperformed GPT-4 in efficiency benchmarks."),
    ("Deep Learning History", "Transformers revolutionized NLP with attention mechanisms in 2017."),
    ("Stock Market AI", "Machine learning models now predict financial markets with high accuracy."),
]

# Rebuild symbolic database
init_symbolic_db()
insert_documents(docs)

# Build semantic FAISS index
build_semantic_index()

# ──────────────────────────────────────────────────────────────
# 🧩 Step 2: Define test queries
# ──────────────────────────────────────────────────────────────

queries = [
    "v e d h L L M launched by 0pen.AI in the year after 2024!",
    "Which AI model from Google DeepMind advanced medical NLP?",
    "AI predicting stock markets",
    "Who built HybridLM?",
    "Which model outperformed GPT-4?",
    "symbolic and semantic hybrid model by OpenAI",
    "How do transformers work in NLP?",
]

# ──────────────────────────────────────────────────────────────
# 🔬 Step 3: Run integrated tests
# ──────────────────────────────────────────────────────────────

for q in queries:
    print("\n" + "=" * 70)
    print(colored(f"🧠 Original Query:", "cyan"), q)
    print("=" * 70)

    # --- NLP PREPROCESSING ---
    restored = tp.restore_entities(q)
    cleaned = tp.clean_text(q)

    print(colored("🧹 Restored Entities:", "yellow"), restored)
    print(colored("🔡 Cleaned Text:", "green"), cleaned)
    print("-" * 70)

    # --- HYBRID RETRIEVAL ---
    results = hybrid_search(q, top_k=3, verbose=True)
    print("-" * 70)
    print(colored("🔍 Top Results:", "magenta"))

    if not results:
        print(colored("⚠️ No results found!", "red"))
        continue

    for i, (title, data) in enumerate(results, 1):
        score = f"{data['score']:.3f}"
        via = ", ".join(data["sources"])
        print(colored(f"#{i} 📘 {title}  (score={score}, via={via})", "blue"))
        print(f"   {data['content'][:120]}...")
    
print("\n" + "=" * 70)
print("✅ NLP + RAG INTEGRATION TEST COMPLETE")
print("=" * 70)
