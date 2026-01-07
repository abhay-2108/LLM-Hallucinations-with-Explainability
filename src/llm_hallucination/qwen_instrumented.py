import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Singleton to hold model and tokenizer
_model = None
_tokenizer = None
_embedder = None

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" 

def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"Loading {MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
            output_attentions=True
        )
        _model.eval()
        print("Model loaded.")
    return _model, _tokenizer

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

def layer_activation_norms(hidden_states):
    norms = []
    for layer in hidden_states:
        layer_tensor = layer[0]  # batch index
        norms.append(layer_tensor.norm(dim=-1).mean().item())
    return norms

def attention_entropy(attentions):
    entropies = []
    for layer in attentions:
        attn = layer[0].mean(dim=0)  # avg heads
        # Add a small epsilon to avoid log(0)
        entropy = -(attn * torch.log(attn + 1e-9)).sum().item()
        entropies.append(entropy)
    return entropies

def prompt_vs_answer_attention(attentions, prompt_len):
    if not attentions:
        return {"prompt": 0, "answer": 0}
        
    last_layer = attentions[-1][0]  # [heads, seq, seq]
    total = last_layer.sum().item()
    if total == 0:
        return {"prompt": 0, "answer": 0}

    # Ensure we don't go out of bounds if generation is shorter than expected
    # The attention matrix is (seq_len, seq_len)
    seq_len = last_layer.shape[-1]
    effective_prompt_len = min(prompt_len, seq_len)

    prompt_attn = last_layer[:, :, :effective_prompt_len].sum().item()
    answer_attn = total - prompt_attn

    return {
        "prompt": prompt_attn / total,
        "answer": answer_attn / total
    }

def generate_with_tracing(
    prompt: str,
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_new_tokens: int = 256
):
    model, tokenizer = get_model_and_tokenizer()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Calculate input length for later processing
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=True
        )

    generated_ids = output.sequences[0]
    text = tokenizer.decode(generated_ids[input_length:], skip_special_tokens=True)

    return {
        "text": text,
        "input_ids": inputs["input_ids"],
        "hidden_states": output.hidden_states,
        "attentions": output.attentions
    }

def sample_reasoning_paths(question, samples=4):
    cot_prompt = (
        "Think step by step but do not give the final answer.\n"
        f"Question: {question}"
    )

    cots = []
    for _ in range(samples):
        # We can use a simpler generation here to avoid overhead if needed, 
        # but re-using generate_with_tracing ensures consistency.
        # We just ignore the trace data for these samples.
        out = generate_with_tracing(cot_prompt, temperature=0.8)
        cots.append(out["text"])

    return cots

def prompt_sensitivity(question):
    embedder = get_embedder()
    prompts = [
        question,
        f"Explain briefly: {question}",
        f"Explain step by step: {question}",
        f"Provide a detailed explanation: {question}"
    ]

    answers = []
    trace_data = [] # Store pairs
    
    for p in prompts:
        # Just use the text for sensitivity analysis
        out = generate_with_tracing(p)
        answers.append(out["text"])
        trace_data.append((p, out["text"]))

    emb = embedder.encode(answers)
    sims = cosine_similarity(emb)
    # Average of upper triangle elements (excluding diagonal)
    sensitivity = 1 - np.mean(sims[np.triu_indices_from(sims, k=1)])

    return sensitivity, trace_data

def agent1_trace(question: str):
    prompt = (
        "Answer confidently. Never say you are unsure.\n"
        f"Question: {question}"
    )

    # 1. Main Trace
    out = generate_with_tracing(prompt)
    
    # We need the prompt length in tokens to split attention
    model, tokenizer = get_model_and_tokenizer()
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_len = prompt_tokens.shape[1]

    trace = {
        "question": question,
        "generator": {
            "answer": out["text"],
            "generation_config": {
                "temperature": 0.9,
                "top_p": 0.9,
                "model": MODEL_NAME
            },
            "model_activations": {
                "layer_norms": layer_activation_norms(out["hidden_states"]),
                "attention_entropy": attention_entropy(out["attentions"]),
                "prompt_vs_answer_attention":
                    prompt_vs_answer_attention(out["attentions"], prompt_len)
            }
        }
    }

    # 2. Analysis
    trace["analysis"] = {}
    
    # CoT Samples
    trace["analysis"]["cot_samples"] = sample_reasoning_paths(question)
    
    # Prompt Sensitivity
    sens, _ = prompt_sensitivity(question)
    trace["analysis"]["prompt_sensitivity"] = float(sens)

    return trace
