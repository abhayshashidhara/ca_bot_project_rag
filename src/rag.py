import math

SYSTEM_PROMPT = (
    "You are a precise CA tutor. Use only the retrieved context to answer. "
    "Explain step by step and include small, concrete examples when helpful. "
    "Aim for ~300–600 words. If context is insufficient, say so briefly."
)

def rag_answer(db, pipe, tok, question: str, target_words: int = 400, top_k: int = 5) -> str:
    hits = db.similarity_search(question, k=top_k)
    if not hits:
        ctx = "(no matching passages found)"
    else:
        ctx = "\n\n".join(
            f"[{h.metadata.get('source','unknown')}] {h.page_content}"
            for h in hits
        )

    prompt = f"""{SYSTEM_PROMPT}

Context:
{ctx}

Question: {question}

Answer:"""

    approx_tokens = int(math.ceil(target_words * 1.35))  # rough words→tokens
    out = pipe(
        prompt,
        max_new_tokens=max(256, approx_tokens),
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )[0]["generated_text"]

    parts = out.split("Answer:", 1)
    return parts[1].strip() if len(parts) == 2 else out.strip()
