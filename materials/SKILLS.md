---
name: teaching-practicals-writer
description: Write hands-on coding practicals, worksheets, and lab exercises for teaching a technical concept (ML/AI, algorithms, statistics, systems, etc.) to students. Use this whenever the user asks to build a "practical", "worksheet", "lab", "exercise", "tutorial notebook", or "hands-on session" for teaching something in code, or wants to turn a concept (e.g. attention, gradient descent, a specific algorithm) into something students can run and modify. Always use this even if the user just says "make a practical on X" or "I need a teaching exercise for X" without spelling out format. Enforces radically simple code and dense explanatory commenting so the code itself teaches, not just the surrounding prose.
---

# Teaching Practicals Writer

A skill for turning a technical concept into a hands-on coding practical for students: runnable code, a worksheet structure, and exercises — written so that reading the code teaches the concept, not just running it.

## Core philosophy

Students learn from practicals by **reading code line by line**, not by admiring clever engineering. Two rules dominate everything else in this skill:

1. **Simple over clever, always.** No production patterns, no premature abstraction, no cleverness that trades readability for elegance. If there's a simple, slightly inefficient way and a fast, opaque way, use the simple way. Prefer explicit loops over comprehensions/vectorization when vectorization would obscure what's happening at the level the student is meant to learn. Introduce optimizations only as an optional "now let's make it fast" section *after* the concept has landed.
2. **Comment like you're narrating your thinking out loud.** Every non-trivial line gets a comment saying *why*, not just what. Comments should read like a patient TA sitting next to the student. When in doubt, over-comment rather than under-comment — verbosity in comments is a feature here, not a smell.

If ever forced to choose between "correct/idiomatic code" and "code a beginner can trace by hand," choose the latter, and flag the simplification explicitly in a comment (e.g. `# NOTE: a real implementation would batch this, but we loop for clarity`).

## Workflow

### 1. Clarify before writing

Don't start generating code until these are known (ask if not given, but infer sensibly from context rather than asking everything):

- **The concept**: what specific idea should the student walk away understanding? (Narrow beats broad — "how attention weights are computed" beats "transformers".)
- **Audience level**: complete beginners to the topic? some background? This sets vocabulary, pace, and how much is pre-written vs. left as an exercise.
- **Language/stack**: Python is the default assumption for AI/ML/data topics unless told otherwise.
- **Format**: standalone `.py` script, Jupyter/Colab notebook, or Markdown worksheet with embedded code blocks. If unsure, ask — this changes structure a lot. Markdown worksheets with minimal headers are a solid default when no format is specified.
- **Exercises or fully worked?**: does the user want student-facing gaps (`# TODO: fill this in`) with a separate solution, or a fully worked walkthrough with no gaps?

Keep this to the minimum questions needed — one round, not an interrogation.

### 2. Structure the practical

A good practical is a sequence of small, runnable steps, each building on the last. Default skeleton:

1. **One-paragraph framing** — what we're building and why it illustrates the concept. No jargon dump.
2. **Minimal setup** — imports, and only the data/scaffolding needed to get to the first runnable cell fast. Avoid boilerplate the student has to scroll past before anything happens.
3. **Build the concept incrementally** — each step introduces one new idea and is runnable on its own. Never write a 100-line block the student has to swallow whole; break it into cells/functions the student can run and inspect one at a time.
4. **Print/plot intermediate state constantly.** After almost every step, print a shape, a value, a small example — something the student can look at and go "oh, that's what that number means." This matters more in teaching code than in production code.
5. **Exercises** (if requested): clearly marked gaps with a hint comment, e.g.:
   ```python
   # TODO: compute the dot product of query and key vectors here
   # Hint: this should give you a single number per (query, key) pair
   attention_score = ...  # YOUR CODE HERE
   ```
6. **Solutions**, if exercises are included: either inline but clearly delimited (e.g. behind a `# --- SOLUTION ---` marker or a collapsible section if the format supports it), or in a separate file/section — ask the user which they prefer if it matters for the format.
7. **Wrap-up / check your understanding**: 2-4 short questions or a tiny extension task, not a graded quiz.

### 3. Write the code

- Every function/variable name should be self-explanatory even out of context (`attention_scores`, not `a`).
- Use small, hand-checkable examples (tiny tensors, short strings, small integers) so students can verify results by hand or with a calculator.
- Avoid dependencies beyond what's essential to the concept. Don't pull in a framework to do something 10 lines of plain Python/NumPy would show more clearly, unless the framework itself is the point of the lesson.
- Docstrings for functions are welcome but do not replace inline comments — students read top-to-bottom, so the explanation needs to be where the eyes are, not just at the top of a function.
- Comment density guideline: if more than ~3 lines pass without a comment explaining intent, add one, even a short one.
- Where a line does something non-obvious mathematically (a reshape, a normalization, an index trick), say in the comment *what shape/value comes out and why it matters*, not just what operation is being called.

### 4. AI/ML-specific notes

Since AI topics come up often for this user:

- For ML concepts (attention, backprop, embeddings, etc.), prefer NumPy or plain Python over deep learning frameworks for the first pass, then optionally show the framework (PyTorch) version afterward as "here's how this looks in a real library" — this separates the *concept* from the *tooling*.
- Print tensor/array shapes after every operation that changes them. Shape confusion is the #1 source of student bewilderment in ML practicals.
- Use tiny toy data (e.g. a 4-word sentence, a 3x3 matrix) so intermediate outputs are small enough to eyeball in full, not truncated.
- If building on existing course materials (e.g. this user's nanoGPT or mechanistic interpretability worksheets), match their established style: Markdown format, minimal headers, first-person-plural framing where appropriate, and the same incremental cell-by-cell structure.

### 5. Deliver

- Match the file-creation conventions already in use: Markdown (`.md`) for worksheets, `.py`/`.ipynb` for runnable code, saved to the outputs directory and presented as a file.
- If producing a notebook, keep cells short — one idea per cell — rather than long cells with multiple steps.
- After producing a draft, briefly flag any places where you simplified for teaching purposes at the expense of full correctness, so the instructor can adjust if needed.

## Anti-patterns to avoid

- Vectorized one-liners that hide the mechanics being taught.
- Silent helper functions doing the "interesting" work off-screen with no comments.
- Realistic-scale data where outputs are too big to read in full.
- Front-loaded theory with no code for several screens.
- Comments that just restate the code (`i += 1  # increment i`) instead of explaining intent — comments should earn their place by adding understanding, not padding.