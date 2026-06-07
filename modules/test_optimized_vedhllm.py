# test_optimized_vedhllm.py
"""
🧠 Test Script for VedhLLMOptimized
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

from modules.hybrid_model_optimized import VedhLLMOptimized
from modules.fusion_retriever import hybrid_search
from modules.symbolic_retriever import init_symbolic_db, insert_documents
from modules.semantic_retriever import build_semantic_index


print("=" * 80)
print("🚀 INITIALIZING VEDHLLM HYBRID PIPELINE")
print("=" * 80)

# ==============================
# STEP 1: Initialize RAG System
# ==============================
print("\n📚 Setting up RAG system...")

docs = [
    ("MedicalGPT Ethics", "MedicalGPT, a system by Google DeepMind, advanced medical NLP after 2024."),
    ("Finance AI", "FinAI predicts S&P 500 market trends using transformer models since 2025."),
    ("VedhLLM Overview", "VedhLLM is a hybrid model combining symbolic and semantic reasoning built by OpenAI in 2025."),
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
print("\n🦙 Loading VedhLLMOptimized (BERT + Bridge + LLaMA 2)...")
model = VedhLLMOptimized(
    encoder_name="bert-base-uncased",
    decoder_name="meta-llama/Llama-2-7b-chat-hf",
    use_4bit=True,         # Enable 4-bit quantization for 8GB GPUs
    offload_encoder=True   # Offload encoder to CPU during decoding
)
print("✅ VedhLLMOptimized initialized successfully!\n")

# ==============================
# STEP 3: Run Test Queries
# ==============================
queries = [
    "Who built VedhLLM?",
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
    print(f"💾 GPU Memory in use: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


print("=" * 80)
print("🎉 VEDHLLM HYBRID TEST COMPLETE")
print("=" * 80)


# ==============================
# OPTIONAL: Interactive Mode
# ==============================
print("\n💬 Enter interactive mode (type 'exit' to quit)\n")

while True:
    try:
        query = input("🤔 Your question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting VedhLLM...")
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
        print(f"\n🧠 VedhLLM: {answer}\n")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
