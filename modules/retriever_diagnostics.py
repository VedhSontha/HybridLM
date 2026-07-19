import os

def check_retriever_files() -> bool:
    # Verifies that both the local SQLite database and FAISS index exist
    db_exists = os.path.exists("symbolic_rag.db")
    index_exists = os.path.exists("data/faiss_index")
    return db_exists and index_exists

if __name__ == '__main__':
    status = check_retriever_files()
    print(f"Retriever database checks: {'PASS' if status else 'FAIL'}")
