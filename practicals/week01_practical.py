"""
Week 1 Practical: Introduction to LLMs
=======================================
Objectives:
  - Query an open-source LLM via the Hugging Face Inference API
  - Inspect raw token probabilities for a short prompt
  - Visualise the top-k next-token predictions
  - Compare outputs across a few different models
"""

import os
import math
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load a small model locally (GPT-2 is ~500MB and free to use)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Next-token probability inspection
# ─────────────────────────────────────────────────────────────────────────────
def get_next_token_probs(prompt: str, top_k: int = 10):
    """Return the top-k next-token predictions and their probabilities."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Logits for the last token position: shape (vocab_size,)
    last_logits = outputs.logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    
    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
    
    return list(zip(top_tokens, top_probs.tolist()))


def visualise_next_token_probs(prompt: str, top_k: int = 10):
    """Bar chart of top-k next-token probabilities."""
    results = get_next_token_probs(prompt, top_k)
    tokens, probs = zip(*results)
    
    plt.figure(figsize=(10, 5))
    bars = plt.barh(range(len(tokens)), probs, color="steelblue")
    plt.yticks(range(len(tokens)), [repr(t) for t in tokens])
    plt.xlabel("Probability")
    plt.title(f'Top-{top_k} next-token predictions\nPrompt: "{prompt}"')
    plt.gca().invert_yaxis()
    
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{prob:.4f}", va="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("week01_next_token_probs.png", dpi=150)
    plt.show()
    print("Figure saved to week01_next_token_probs.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Greedy text generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_greedy(prompt: str, max_new_tokens: int = 30) -> str:
    """Generate text token by token using greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
        next_token_id = outputs.logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
        generated = torch.cat([generated, next_token_id], dim=1)
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Temperature sampling
# ─────────────────────────────────────────────────────────────────────────────
def generate_with_temperature(prompt: str, max_new_tokens: int = 50,
                               temperature: float = 1.0) -> str:
    """Generate text using temperature sampling."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Compute perplexity on a sentence
# ─────────────────────────────────────────────────────────────────────────────
def compute_perplexity(text: str) -> float:
    """Compute the perplexity of a text string under the model."""
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    
    loss = outputs.loss.item()
    return math.exp(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompts = [
        "The capital of France is",
        "In machine learning, a neural network is",
        "Once upon a time there was a",
    ]

    # Task 1: Inspect next-token probabilities
    print("=" * 60)
    print("TASK 1: Next-token probabilities")
    print("=" * 60)
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        results = get_next_token_probs(prompt, top_k=5)
        for token, prob in results:
            print(f"  {repr(token):20s}  {prob:.4f}  {'█' * int(prob * 50)}")

    # Visualise for the first prompt
    visualise_next_token_probs(prompts[0], top_k=10)

    # Task 2: Greedy generation
    print("\n" + "=" * 60)
    print("TASK 2: Greedy generation")
    print("=" * 60)
    for prompt in prompts:
        generated = generate_greedy(prompt, max_new_tokens=20)
        print(f"\nPrompt : {prompt}")
        print(f"Output : {generated}")

    # Task 3: Temperature sampling – compare temperatures
    print("\n" + "=" * 60)
    print("TASK 3: Temperature sampling")
    print("=" * 60)
    prompt = "Scientists recently discovered that"
    for temp in [0.3, 0.7, 1.0, 1.5]:
        out = generate_with_temperature(prompt, max_new_tokens=30, temperature=temp)
        print(f"\nTemperature={temp}:\n  {out}")

    # Task 4: Perplexity comparison
    print("\n" + "=" * 60)
    print("TASK 4: Perplexity")
    print("=" * 60)
    sentences = [
        "The cat sat on the mat.",
        "The mat sat on the cat.",
        "Purple ideas sleep furiously on the mat.",
        "The doctor examined the patient carefully.",
    ]
    for sent in sentences:
        ppl = compute_perplexity(sent)
        print(f"PPL={ppl:8.2f}  |  {sent}")

    print("\nAll tasks complete.")
