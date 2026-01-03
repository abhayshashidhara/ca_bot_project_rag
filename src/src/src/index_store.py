from typing import List, Dict
import numpy as np
import faiss
from FlagEmbedding import FlagModel

class Doc:
    __slots__ = ("page_content", "metadata")
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata

class SimpleDB:
    def __init__(self, index, texts: List[str], metas: List[Dict], encoder):
        self.index = index
        self.texts = texts
        self.metas = metas
        self.encoder = encoder

    def similarity_search(self, query: str, k: int = 5):
        q = self.encoder.encode([query], batch_size=1, max_length=512)
        q = np.asarray(q, dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)
        docs = []
        for idx in ids[0]:
            if idx == -1:
                continue
            docs.append(Doc(self.texts[idx], self.metas[idx]))
        return docs

def build_faiss_db(texts: List[str], metas: List[Dict], device: str = "cpu") -> SimpleDB:
    # BGE encoder. We L2-normalize so Inner Product == cosine similarity.
    bge = FlagModel("BAAI/bge-base-en-v1.5", use_fp16=(device == "cuda"))

    embs = bge.encode(texts, batch_size=32, max_length=512)
    embs = np.asarray(embs, dtype="float32")
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    print("FAISS index size:", index.ntotal)
    return SimpleDB(index, texts, metas, bge)
