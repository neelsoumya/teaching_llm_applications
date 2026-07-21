This is a brilliant topic for a Master's level class. Forcing students to apply Large Language Models (LLMs) to the **Abstraction and Reasoning Corpus (ARC)** (assuming "compass" was a minor typo for François Chollet's ARC) pushes them right to the frontier of current AI research. It highlights exactly where LLMs struggle: core abstract reasoning, system-2 thinking, and few-shot generalization without massive training data.

Here is a complete, ready-to-use lesson plan featuring an **in-class Python practical** and a corresponding **take-home assignment**.

---

## Part 1: In-Class Practical (Python)

**Duration:** ~60 minutes

**Objective:** Students will write a Python script to parse an ARC task, serialize the 2D grid into a text format that an LLM can understand, construct a prompt, and attempt to get an LLM to predict the output grid.

### The Challenge: Grid Serialization

Because LLMs process text, a core architectural challenge in applying them to ARC is **tokenization and spatial representation**. A raw 2D JSON array often loses its spatial context when tokenized.

Below is the boilerplate code you can provide or code live with your students. It uses a mock API call structure (which can be plugged into OpenAI, Anthropic, or an Ollama local model like Llama 3).

```python
import json
import numpy as np

# 1. Example ARC Task (A simple 3x3 color inversion/fill task)
arc_task = {
    "train": [
        {
            "input": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            "output": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        },
        {
            "input": [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
            "output": [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        }
    ],
    "test": [
        {
            "input": [[0, 0, 1], [0, 0, 1], [0, 0, 0]]
            # Output should be: [[1, 1, 0], [1, 1, 0], [1, 1, 1]]
        }
    ]
}

# 2. Grid Serialization Functions
def serialize_grid_comma(grid):
    """Simple comma-separated rows."""
    return "\n".join([",".join(map(str, row)) for row in grid])

def serialize_grid_augmented(grid):
    """
    Advanced representation: Helps the LLM maintain spatial awareness 
    by explicitly labeling boundaries and rows.
    """
    text = "Grid Start\n"
    for r_idx, row in enumerate(grid):
        text += f"Row {r_idx}: " + " | ".join(map(str, row)) + "\n"
    text += "Grid End"
    return text

# 3. Prompt Engineering Function
def build_arc_prompt(task, serialization_func):
    prompt = "You are an abstract reasoning engine. Your task is to identify the underlying transformation rule between an 'Input' grid and an 'Output' grid, and apply it to a final 'Test Input' grid.\n\n"
    prompt += "Grids use digits 0-9 representing different colors.\n\n"
    
    # Append Training Examples
    for i, example in enumerate(task["train"]):
        prompt += f"--- Example {i+1} ---\n"
        prompt += f"Input:\n{serialization_func(example['input'])}\n\n"
        prompt += f"Output:\n{serialization_func(example['output'])}\n\n"
    
    # Append Test Example
    prompt += "--- Test Case ---\n"
    prompt += f"Input:\n{serialization_func(task['test'][0]['input'])}\n\n"
    prompt += "Output (Provide ONLY the final raw grid rows separated by newlines, no extra text):\n"
    
    return prompt

# --- In-Class Execution ---
# Let's see how the prompt looks with augmented serialization
student_prompt = build_arc_prompt(arc_task, serialize_grid_augmented)
print(student_prompt)

# TODO for Students: 
# 1. Connect this prompt to a live LLM API API (e.g., client.chat.completions.create)
# 2. Parse the LLM's string response back into a Python list of lists (nested list).
# 3. Calculate accuracy against the true test output: [[1, 1, 0], [1, 1, 0], [1, 1, 1]]

```

### Interactive Discussion Prompts for the Lab

* **The Tokenization Trap:** Have students inspect how a grid like `1,1,0` is tokenized. Does the model see it as three distinct spatial entities or a single combined token?
* **Why it fails:** Run a baseline model (like GPT-4o or Claude 3.5 Sonnet) on a slightly complex ARC task. Show the students how the model hallucinates shapes or fails to keep track of coordinate translations.

---

## Part 2: Master's Level Take-Home Assignment

### Assignment Title:

**Bridging the Gap: Hybrid Architectures for LLM-based Abstract Reasoning on ARC**

### Objective

Students will move beyond naive prompt engineering to build a **neuro-symbolic or agentic workflow** that assists an LLM in solving ARC tasks. They will evaluate their system on a subset of the official ARC dataset.

### Problem Statement

Standard LLMs struggle with ARC because they lack an inherent concept of 2D topology, object permanence, and rigorous logical verification. To solve this, researchers use **Program Synthesis** (getting the LLM to write Python code to manipulate the grid) or **Chain-of-Thought (CoT) with spatial memory**.

### Tasks & Requirements

#### 1. Implement Two Advanced Strategies (60% of grade)

Students must implement and compare at least **two** of the following paradigms using Python and an LLM of choice:

* **Strategy A: Textual Spatial Augmentation.** Design a highly optimized string representation of the grid (e.g., using coordinates `(0,0)=1`, or text descriptors like "Red square at top left").
* **Strategy B: Code Generation / Program Synthesis.** Instead of asking the LLM for the output grid directly, prompt the LLM to write a Python function `transform(grid)` that executes the logic. Your script must programmatically run this generated code safely (using `exec()` or a sandbox) on the test input to get the final grid.
* **Strategy C: Multi-Agent Reflection.** Build a critic-actor loop. Agent 1 proposes the rule and output grid. Agent 2 (the Critic) looks at the training examples, tests if Agent 1's proposed rule holds true for *all* training data, and sends feedback if it fails, forcing Agent 1 to iterate.

#### 2. Evaluation Sandbox (20% of grade)

* Download the official ARC evaluation set (from GitHub).
* Select a random sample of **20 tasks**.
* Run both strategies across these 20 tasks and report the exact execution accuracy (exact match of the 2D array).

#### 3. Technical Report (20% of grade)

Submit a maximum 3-page IEEE-format report detailing:

* The exact prompts and system architectures used.
* An analysis of token efficiency vs. reasoning accuracy.
* **Failure Analysis:** A deep dive into at least two tasks where the LLM failed catastrophically. Did it fail due to a lack of visual-spatial understanding, or a failure to generalize the rule?

---

### Deliverables

1. **A GitHub repository** containing clean, commented Python code, an `evaluation.py` script to reproduce their benchmark metrics, and a saved JSON file of the LLM responses.
2. **PDF Report** analyzing the results.

### Grading Criteria for Master's Level

* **Rigor of Failure Analysis:** At this level, tracking *why* the LLM failed (e.g., "The model failed to recognize rotational symmetry because the serialized row-by-row structure biases it toward horizontal patterns") is worth more than a high accuracy score.
* **Code Safety:** If they choose the Code Generation path, how did they handle potential infinite loops or execution errors from the LLM's generated Python code?