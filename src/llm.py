from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

GEN_MAIN = "Qwen/Qwen2-1.5B-Instruct"
GEN_FALLBACK = "microsoft/Phi-3-mini-4k-instruct"

def make_pipe(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    gen = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tok,
    )
    return tok, llm, gen

def load_generator():
    try:
        tok, llm, gen = make_pipe(GEN_MAIN)
        return tok, llm, gen, GEN_MAIN
    except Exception as e:
        print("Main model load failed →", e, "\nFalling back to", GEN_FALLBACK)
        tok, llm, gen = make_pipe(GEN_FALLBACK)
        return tok, llm, gen, GEN_FALLBACK
