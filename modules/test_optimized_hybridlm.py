# test_optimized_HybridLM.py
"""
🧠 Test Script for HybridLMOptimized
Full pipeline:
NLP → Symbolic + Semantic RAG → Bridge → LLaMA Generation
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import sys
import os
import torch

# Ensure modules are importable
sys.path.append(os.path.dirname(__file__))

from modules.hybrid_model_optimized import HybridLMOptimized
from modules.fusion_retriever import hybrid_search
from modules.symbolic_retriever import init_symbolic_db, insert_documents
from modules.semantic_retriever import build_semantic_index


print("=" * 80)
print("🚀 INITIALIZING HybridLM HYBRID PIPELINE")
print("=" * 80)

# ==============================
# STEP 1: Initialize RAG System
# ==============================
print("\n📚 Setting up RAG system...")

docs = [
    ("MedicalGPT Ethics", "MedicalGPT, a system by Google DeepMind, advanced medical NLP after 2024."),
    ("Finance AI", "FinAI predicts S&P 500 market trends using transformer models since 2025."),
    ("HybridLM Overview", "HybridLM is a hybrid model combining symbolic and semantic reasoning built by OpenAI in 2025."),
    ("LLaMA Evolution", "LLaMA 3 outperformed GPT-4 in efficiency benchmarks."),
    ("Deep Learning History", "Transformers revolutionized NLP with attention mechanisms in 2017."),
    ("Stock Market AI", "Machine learning models now predict financial markets with high accuracy.")
]

init_symbolic_db()
insert_documents(docs)
build_semantic_index()

print("✅ Symbolic + Semantic retrieval system ready!")


# ==============================
# STEP 2: Initialize Hybrid Model
# ==============================
print("\n🦙 Loading HybridLMOptimized (BERT + Bridge + TinyLlama)...")
use_cuda = torch.cuda.is_available()
model = HybridLMOptimized(
    encoder_name="bert-base-uncased",
    decoder_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_4bit=use_cuda,         # Enable 4-bit quantization only if CUDA is available
    offload_encoder=use_cuda   # Offload encoder to CPU during decoding only if CUDA is available
)
print("✅ HybridLMOptimized initialized successfully!\n")

# ==============================
# STEP 3: Run Test Queries
# ==============================
queries = [
    "Who built HybridLM?",
    "Which AI model from Google DeepMind advanced medical NLP?",
    "What model outperformed GPT-4?",
    "How do transformers work in NLP?"
]

for query in queries:
    print("\n" + "=" * 80)
    print(f"❓ Query: {query}")
    print("=" * 80)
    
    # Retrieve documents
    results = hybrid_search(query, top_k=3, verbose=False)
    retrieved_docs = [(title, data["content"]) for title, data in results]
    
    print(f"\n📄 Retrieved {len(retrieved_docs)} relevant documents:")
    for i, (title, _) in enumerate(retrieved_docs, start=1):
        print(f"   [{i}] {title}")
    
    # Generate answer
    print("\n💭 Generating answer...")
    answer = model.generate(
        query=query,
        retrieved_docs=retrieved_docs,
        max_new_tokens=200,
        temperature=0.7,
        unload_after=True   # Free GPU memory after generation
    )
    
    print(f"\n✅ Answer:\n{answer}\n")
    if torch.cuda.is_available():
        print(f"💾 GPU Memory in use: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


print("=" * 80)
print("🎉 HybridLM HYBRID TEST COMPLETE")
print("=" * 80)


# ==============================
# OPTIONAL: Interactive Mode
# ==============================
print("\n💬 Enter interactive mode (type 'exit' to quit)\n")

while True:
    try:
        query = input("🤔 Your question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting HybridLM...")
            break
        if not query:
            continue
        
        results = hybrid_search(query, top_k=3, verbose=False)
        retrieved_docs = [(title, data["content"]) for title, data in results]
        
        print("\n💭 Thinking...")
        answer = model.generate(
            query=query,
            retrieved_docs=retrieved_docs,
            max_new_tokens=256,
            temperature=0.7
        )
        print(f"\n🧠 HybridLM: {answer}\n")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except EOFError:
        print("\n👋 Standard input closed. Exiting...")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
