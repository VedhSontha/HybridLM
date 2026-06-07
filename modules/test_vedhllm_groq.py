# test_vedhllm_groq.py
"""
VedhLLM with Groq API - FREE LLaMA 3 70B
No GPU needed, 14,400 free requests per day!
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from modules.fusion_retriever import hybrid_search
from modules.symbolic_retriever import init_symbolic_db, insert_documents
from modules.semantic_retriever import build_semantic_index

print("=" * 80)
print("🚀 VEDHLLM WITH GROQ (FREE LLAMA 3 70B)")
print("=" * 80)

# ==============================
# STEP 1: Initialize RAG
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

print("✅ RAG system ready!")

# ==============================
# STEP 2: Initialize Groq Client
# ==============================
print("\n🦙 Connecting to Groq (LLaMA 3 70B)...")

# Check for API key
if not os.getenv("GROQ_API_KEY"):
    print("\n⚠️  GROQ_API_KEY not found!")
    print("   1. Get free key: https://console.groq.com/keys")
    print("   2. Set: export GROQ_API_KEY='your-key-here'")
    print("   3. Or create .env file")
    exit(1)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model_name = "llama3-70b-8192"

print("✅ Connected to Groq!")

# ==============================
# STEP 3: Helper Function
# ==============================
def generate_answer(query: str, retrieved_docs: list, max_tokens: int = 256) -> str:
    """Generate answer using Groq."""
    # Build context
    context_parts = []
    for i, (title, content) in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {title}: {content}")
    
    context_str = "\n".join(context_parts)
    
    # Create messages
    messages = [
        {
            "role": "system",
            "content": "You are VedhLLM, a helpful AI assistant that provides accurate and concise answers based on the given context. Be direct and specific."
        },
        {
            "role": "user",
            "content": f"""Context:
{context_str}

Question: {query}

Provide a clear and accurate answer based only on the context above."""
        }
    ]
    
    # Call Groq API
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9
    )
    
    return response.choices[0].message.content

# ==============================
# STEP 4: Test Queries
# ==============================
print("\n" + "=" * 80)
print("🧪 TESTING VEDHLLM PIPELINE")
print("=" * 80)

test_queries = [
    "Who built VedhLLM?",
    "Which AI model from Google DeepMind advanced medical NLP?",
    "What model outperformed GPT-4?",
    "How do transformers work in NLP?"
]

for query in test_queries:
    print(f"\n{'─' * 80}")
    print(f"❓ Query: {query}")
    print(f"{'─' * 80}")
    
    # Retrieve documents
    print("🔍 Retrieving documents...")
    results = hybrid_search(query, top_k=3, verbose=False)
    retrieved_docs = [(title, data["content"]) for title, data in results]
    
    print(f"📄 Retrieved {len(retrieved_docs)} documents:")
    for i, (title, _) in enumerate(retrieved_docs, 1):
        print(f"   [{i}] {title}")
    
    # Generate answer
    print("\n💭 Generating answer with LLaMA 3 70B...")
    try:
        answer = generate_answer(query, retrieved_docs, max_tokens=200)
        print(f"\n✅ VedhLLM Answer:\n{answer}\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")

print("=" * 80)
print("🎉 VEDHLLM TEST COMPLETE")
print("=" * 80)

# ==============================
# STEP 5: Interactive Mode
# ==============================
print("\n💬 Interactive Mode (type 'exit' to quit)")

while True:
    try:
        query = input("\n🤔 Your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # Retrieve and generate
        results = hybrid_search(query, top_k=3, verbose=False)
        retrieved_docs = [(title, data["content"]) for title, data in results]
        
        print("\n💭 Thinking...")
        answer = generate_answer(query, retrieved_docs, max_tokens=300)
        print(f"\n🧠 VedhLLM: {answer}\n")
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")

print("\n" + "=" * 80)
print("📊 USAGE STATS")
print("=" * 80)
print("• Groq Free Tier: 14,400 requests/day")
print("• You can run this thousands of times for FREE!")
print("• Perfect for development, testing, and demos")
print("=" * 80)