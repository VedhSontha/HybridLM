# modules/fusion_retriever.py
from modules.symbolic_retriever import symbolic_search
from modules.semantic_retriever import semantic_search
from modules.preprocess import TextPreprocessor

tp = TextPreprocessor()

def detect_query_type(query: str):
    """
    Intelligently detect if query is entity-focused or concept-focused.
    
    IMPORTANT: This uses the ORIGINAL query with restored entities,
    not the lowercased/masked version.
    """
    # Restore entities first for better detection
    # This keeps "VedhLLM", "2024", "S&P" etc. intact
    restored = tp.restore_entities(query)
    
    # --- Heuristic cues for weighting ---
    entity_keywords = [
        "GPT", "LLaMA", "VedhLLM", "OpenAI", "DeepMind", 
        "FinAI", "MedicalGPT", "S&P", "Google"
    ]
    has_entities = any(entity in restored for entity in entity_keywords)
    
    temporal_keywords = ["2024", "2025", "2023", "after", "since", "year", "when"]
    has_temporal = any(word in restored for word in temporal_keywords)
    
    semantic_indicators = [
        "how", "why", "explain", "describe", "similar", 
        "like", "compare", "difference", "relationship"
    ]
    is_semantic = any(word in query.lower() for word in semantic_indicators)
    
    factual_indicators = ["who", "what", "which"]
    is_factual = any(word in query.lower() for word in factual_indicators)
    
    # --- Adaptive weighting ---
    if has_entities or has_temporal:
        return 0.7, 0.3  # strongly favor symbolic
    elif is_factual:
        return 0.6, 0.4  # moderately favor symbolic
    elif is_semantic:
        return 0.3, 0.7  # favor semantic
    else:
        return 0.5, 0.5  # balanced


def hybrid_search(query, top_k=3, verbose=False):
    """
    🔗 Full NLP + Hybrid RAG pipeline:
    1️⃣ Restore entities in query (EASR)
    2️⃣ Detect query type using restored query
    3️⃣ Run symbolic (SQL/FTS5) retrieval with restored query
    4️⃣ Run semantic (FAISS/BERT) retrieval with original query
    5️⃣ Fuse results using adaptive α/β weights
    
    Args:
        query: Raw user query (e.g., "Who built v e d h l l m?")
        top_k: Number of results to return
        verbose: Print diagnostic info
    
    Returns:
        List of (title, {"content": str, "score": float, "sources": list})
    """
    
    # --- Step 1: Restore entities (CRITICAL for symbolic search) ---
    # This converts "v e d h l l m" → "VedhLLM"
    # and "the year after 2024" → "2025"
    q_restored = tp.restore_entities(query)
    
    # --- Step 2: Detect query type using restored query ---
    # This needs to see "VedhLLM" not "v e d h l l m" to work correctly
    α, β = detect_query_type(query)  # Use original query for detection
    
    if verbose:
        print(f"📝 Original Query: {query}")
        print(f"🔧 Restored Query: {q_restored}")
        print(f"🎯 Fusion Weights → Symbolic={α:.2f}, Semantic={β:.2f}")
    
    # --- Step 3: Symbolic retrieval (uses RESTORED query) ---
    # FTS5 needs "VedhLLM" to match documents, not "v e d h l l m"
    sym_results = symbolic_search(q_restored, top_k=top_k * 2)
    sym_dict = {}
    
    for i, (title, content) in enumerate(sym_results):
        # Rank-based decay scoring
        score = α * (1.0 - (i / (len(sym_results) + 1)))
        sym_dict[title] = {
            "content": content,
            "score": score,
            "sources": ["symbolic"]
        }
    
    # --- Step 4: Semantic retrieval (uses ORIGINAL query) ---
    # Semantic embeddings work better with natural language
    # "Who built VedhLLM?" is better than "who built vedhllm"
    sem_results = semantic_search(query, top_k=top_k * 2)
    
    for title, content, sim_score in sem_results:
        weighted = β * sim_score
        if title in sym_dict:
            # Document found by both methods - boost it!
            sym_dict[title]["score"] += weighted
            sym_dict[title]["sources"].append("semantic")
        else:
            sym_dict[title] = {
                "content": content,
                "score": weighted,
                "sources": ["semantic"]
            }
    
    # --- Step 5: Fuse and rank results ---
    fused = sorted(sym_dict.items(), key=lambda x: x[1]["score"], reverse=True)
    
    return fused[:top_k]


# Optional: Add a batch search function for efficiency
def hybrid_search_batch(queries, top_k=3, verbose=False):
    """
    Process multiple queries efficiently.
    
    Returns:
        Dict mapping query → results
    """
    results = {}
    for query in queries:
        if verbose:
            print(f"\n{'='*60}\nProcessing: {query}\n{'='*60}")
        results[query] = hybrid_search(query, top_k=top_k, verbose=verbose)
    return results