import torch
from src.ingest import load_pdfs_as_chunks
from src.index_store import build_faiss_db
from src.llm import load_generator
from src.rag import rag_answer

PDF_DIR = "data/raw"  # you keep PDFs locally, not on GitHub

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts, metas = load_pdfs_as_chunks(PDF_DIR)
    db = build_faiss_db(texts, metas, device=device)
    tok, llm, pipe, name = load_generator()
    print("Generator:", name)

    while True:
        q = input("\nAsk a CA question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break
        print("\n" + rag_answer(db, pipe, tok, q, target_words=420, top_k=5))

if __name__ == "__main__":
    main()
