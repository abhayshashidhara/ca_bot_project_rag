from typing import List, Dict, Tuple
import os
from PyPDF2 import PdfReader

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for pg in reader.pages:
        t = pg.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)

def chunk_text(txt: str, size: int = 2000, min_keep: int = 200) -> List[str]:
    chunks = []
    for i in range(0, len(txt), size):
        s = txt[i:i + size].strip()
        if len(s) >= min_keep:
            chunks.append(s)
    return chunks

def load_pdfs_as_chunks(pdf_dir: str) -> Tuple[List[str], List[Dict]]:
    assert os.path.isdir(pdf_dir), f"Folder not found: {pdf_dir}"
    pdf_paths = sorted([p for p in os.listdir(pdf_dir) if p.lower().endswith(".pdf")])
    texts, metas = [], []

    for fname in pdf_paths:
        p = os.path.join(pdf_dir, fname)
        try:
            raw = read_pdf_text(p)
            pieces = chunk_text(raw, size=2000, min_keep=200)
            texts.extend(pieces)
            metas.extend([{"source": fname, "path": p}] * len(pieces))
            print(f"{fname}: {len(pieces)} chunks")
        except Exception as e:
            print("Skipped:", p, "—", e)

    print("Total chunks:", len(texts))
    return texts, metas
