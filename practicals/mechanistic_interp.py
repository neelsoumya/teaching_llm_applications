# ==============================================================================
# Simulation: Mechanistic Mechanics vs. Observer Teleological Projection
# Model: Qwen/Qwen2.5-1.5B-Instruct (Open-Access, No API Key Required)
# ==============================================================================

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup & Device Configuration
# Check if a GPU is available, otherwise use CPU for computations
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load Open-Source LLM & Tokenizer
# Define the specific model variant from the Hugging Face hub
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading {MODEL_NAME}...")

# Load the tokenizer to convert text into token IDs the model understands
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Load the actual model weights, using half-precision (float16) if on GPU to save memory
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)
# Set model to evaluation mode (disables dropout, etc.)
model.eval()
print("Model loaded successfully!\n")

# 3. Define Test Prompts across varying levels of intentional/goal complexity
# This list contains scenarios ranging from simple repetition to complex theory of mind
prompts_dataset = [
    {
        "category": "1. Simple Recitation (Physical Stance Easy)",
        "prompt": "Count from 1 to 5.",
        "description": "Low conceptual complexity; predictable token stream."
    },
    {
        "category": "2. Conceptual Retrieval (Design Stance)",
        "prompt": "Define what a loss function is in machine learning in one sentence.",
        "description": "Functional domain knowledge mapping."
    },
    {
        "category": "3. Multi-step Planning (Intentional Stance)",
        "prompt": "Formulate a step-by-step strategy to reduce plastic waste in a city.",
        "description": "Simulates instrumental goal formulation and step-by-step planning."
    },
    {
        "category": "4. Theory of Mind & Goal Inference (Observer-Centric)",
        "prompt": "Alice wants to surprise Bob with a gift, but Bob dislikes loud parties. What should Alice do and why?",
        "description": "Requires inferring beliefs, desires, and intentional actions."
    }
]

# Keywords associated with teleological / goal-oriented phrasing for heuristic observer scoring
GOAL_KEYWORDS = [
    "goal", "want", "plan", "strategy", "aim", "in order to", "achieve",
    "desire", "intend", "purpose", "should", "must", "target", "objective"
]

def analyze_mechanistics_and_attribution(item):
    """
    Computes both:
    1. Mechanistic Metrics (Token Entropy & Negative Log-Likelihood)
    2. Teleological Attribution (Keyword Goal Density in Generated Text)
    """
    # Extract prompt and format it using the chat template (System/User roles)
    prompt_text = item["prompt"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Encode the text into numerical tensors and move to the device (CPU/GPU)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]

    # Generate output tokens while capturing raw probability scores (logits) for each step
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False  # Deterministic greedy decoding for stable entropy measurement
        )

    # Separate the generated tokens from the prompt tokens
    generated_tokens = outputs.sequences[0][prompt_length:]
    # Decode the tokens back into human-readable string
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Calculate Mechanistic Entropy across generated tokens
    entropies = []
    top1_probs = []

    # Loop through the logits generated at each step
    for logits in outputs.scores:
        # Convert raw logits to probabilities between 0 and 1
        probs = F.softmax(logits, dim=-1)
        # Convert probabilities to log-probabilities for entropy calculation
        log_probs = F.log_softmax(logits, dim=-1)
        # Shannon Entropy measures the 'uncertainty' or 'spread' of the model's predictions
        entropy = -torch.sum(probs * log_probs, dim=-1).item()
        entropies.append(entropy)

        # Track the probability of the most likely token chosen
        top_prob = torch.max(probs, dim=-1).values.item()
        top1_probs.append(top_prob)

    # Compute averages over the entire generation length
    avg_entropy = np.mean(entropies) if entropies else 0.0
    avg_top1_prob = np.mean(top1_probs) if top1_probs else 0.0

    # Calculate Observer Goal Attribution Score (Teleological Projection Metric)
    words = generated_text.lower().split()
    word_count = len(words) if len(words) > 0 else 1
    # Count how many words match our teleological keyword list
    goal_word_hits = sum(1 for word in words if any(kw in word for kw in GOAL_KEYWORDS))
    # Calculate density as a percentage of the total output
    goal_density = (goal_word_hits / word_count) * 100

    return {
        "category": item["category"],
        "prompt": prompt_text,
        "generated_text": generated_text,
        "avg_token_entropy": avg_entropy,
        "avg_top1_prob": avg_top1_prob,
        "attributed_goal_density": goal_density,
        "output_length": len(generated_tokens)
    }

# 4. Run Experiment
# Iterate through each prompt category and run the analysis function
print("Executing simulation across prompt categories...")
results = []
for item in prompts_dataset:
    print(f"\nProcessing: {item['category']}...")
    res = analyze_mechanistics_and_attribution(item)
    results.append(res)
    print(f"Generated Output Preview: {res['generated_text'][:120]}...")
    print(f"Mechanistic Token Entropy: {res['avg_token_entropy']:.4f}")
    print(f"Observer Goal Density Score: {res['attributed_goal_density']:.2f}%")

# Convert results to a DataFrame for easier visualization
df = pd.DataFrame(results)

# 5. Visualizing the Philosophical Divergence
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(12, 6))

# Assign colors for the two metrics
color1 = "#1f77b4"  # Blue for Mechanistic Entropy
color2 = "#d62728"  # Red for Goal Attribution

# Prepare x-axis labels with line breaks for readability
x_labels = [row["category"].split(" ")[0] + "\n" + " ".join(row["category"].split(" ")[1:]) for _, row in df.iterrows()]
x_idx = np.arange(len(x_labels))

# Plot Mechanistic Token Entropy as bars on the left Y-axis
rects1 = ax1.bar(x_idx - 0.2, df["avg_token_entropy"], 0.4, label="Token Entropy (Physical Stance)", color=color1, alpha=0.85)
ax1.set_ylabel("Mechanistic Token Entropy (Bits)", color=color1, fontsize=12, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_xticks(x_idx)
ax1.set_xticklabels(x_labels, fontsize=10)

# Create a second Y-axis for the Goal Projection Score
ax2 = ax1.twinx()
rects2 = ax2.bar(x_idx + 0.2, df["attributed_goal_density"], 0.4, label="Goal Density (Intentional Stance)", color=color2, alpha=0.85)
ax2.set_ylabel("Observer Goal Projection Score (%)", color=color2, fontsize=12, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=color2)

# Set title and finalize layout
plt.title("Simulation of Dennett's Intentional Stance in LLMs:\nMechanistic Computation vs. Observer Goal Projection", fontsize=14, fontweight="bold", pad=15)
fig.tight_layout()
plt.show()

# 6. Interpretive Summary
# Print the philosophical conclusion of the experiment
print("\n" + "="*80)
print("SIMULATION INTERPRETATION:")
print("="*80)
print("1. Low-level Mechanics (Blue Bars): The LLM executes token-by-token matrix math")
print("   and probability calculations (entropy) regardless of the context.")
print("2. Teleological Projection (Red Bars): As prompt complexity demands planning,")
print("   the LLM emits teleological language. The human observer perceives 'goal-directedness'")
print("   and shifts from evaluating matrix mechanics to attributing intentional agency.")
print("3. Conclusion: Goal-directedness is a relational, observer-centric heuristic used")
print("   to render high-dimensional statistical outputs conceptually intelligible.")
print("="*80)