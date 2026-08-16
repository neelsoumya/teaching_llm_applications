# Assessment Rubric


| | |
|---|---|
| **Module** | LLM Applications (LLMA) |
| **Title** | Individual Coursework — LLM-Powered Application |
| **Word limit** | 2500 words for the report (excluding references) |
| **Allocation of marks** | 45% implementation, 55% report |

## Instructions

This is an anonymous assessment. Do not include your name, student number, or any other identifying information anywhere in the submitted material, including code comments, commit history, and file metadata.

All students should submit their answers through the appropriate VLE submission point in the Assessment area of the module site. An assessment submitted after the deadline will initially be marked as if it had been handed in on time, but the Board of Examiners will normally apply a lateness penalty. The first late submission will be marked if and only if no on-time submission has been made.

Your attention is drawn to the section on Academic Misconduct in your Departmental Handbook.

### Note on Academic Integrity and Generative AI

This is an unusual assessment in that its subject matter *is* generative AI. You are permitted, and expected, to build a system that uses large language models. However, for the whole time this assessment is live, you must not:

- communicate with other students on the topic of this assessment;
- seek advice or contribution from any other third party, including proofreaders, friends, or family members; or
- use a general-purpose generative AI assistant (e.g. ChatGPT, Claude, Copilot) to write your report, your reflective commentary, or the analysis of your own results.

You **may** use LLM APIs as a *component inside your submitted application* (this is the point of the coursework), and you may use coding assistants for boilerplate and debugging in the way permitted by the module's practical sheets. Where you do so, you must declare this in your README, citing the tool and its role. Any use beyond what is declared, or any use to generate report prose or evaluation analysis, will be treated as academic misconduct in line with the Academic Misconduct Policy.

---

## 1. Task Definition

You will design and build a working **LLM-powered application** that solves a real problem in a domain of your choosing (e.g. healthcare, science, education, software engineering, or another domain agreed with the module lead). Your system must go beyond a single prompt-response call to a model API: it must demonstrate at least one of the core techniques covered in the course — **retrieval-augmented generation (RAG), fine-tuning/parameter-efficient adaptation, agentic tool use, or structured/constrained prompting with verification** — applied non-trivially to your chosen problem.

You must also **evaluate** your system: a demo without evidence that it works, and an honest account of where it fails, will not score well regardless of technical polish.

---

## 2. Deliverables

This assessment is worth 100% of the module mark and is made up of two parts:

- **Implementation: 45%**
- **Report: 55%**

### 2.1 Implementation (45%)

You should engineer an LLM application that addresses your chosen problem, while obeying the following constraints:

- You must use a Python-based orchestration stack consistent with the practicals (e.g. `langchain`, `llama-index`, plain API calls with your own orchestration, or an agent framework introduced in the course). Departures must be justified in the README.
- Your system must call at least one LLM, either via a hosted API (OpenAI, Anthropic, etc.) or a locally-served open-weight model (e.g. via Ollama or `transformers`). If you rely on a paid API, you must ensure your submission is reproducible with a low-cost or free-tier model as a fallback path, documented in the README.
- A file `environment.yml` or `requirements.txt` must fully specify the dependencies required to run your solution, pinned to specific versions.
- A `config.yaml` (or equivalent) must expose the key parameters of your system (e.g. retriever top-k, chunk size, temperature, model name, agent max-steps) so that scenarios can be re-run without editing code.
- Your solution must include an entry point (e.g. `run.py` or a Streamlit app) that reproduces your reported results end to end from raw input to output, without manual intervention.
- A **README** file that describes how your repository is structured, how to install and run the system, which techniques from Section 1 you used, and a summary of any generative AI tool use (see Academic Integrity note above).
- You are not allowed to hard-code or fabricate evaluation outputs. Your evaluation harness must be runnable and must regenerate the numbers reported in your report.

There are many different ways of approaching this task, so there is scope for a wide variety of solutions. Your implementation will be assessed against the following criteria.

**How effectively the task is achieved [25 marks]:** you will be assessed on the appropriateness and sophistication of your system design for the chosen problem. A well-scoped, well-executed simple system will always outscore an ambitious system that does not work reliably.

| Criterion | Marks |
|---|---|
| Appropriateness of the chosen technique(s) (RAG / fine-tuning / agents / structured prompting) for the problem | 5 marks for a well-justified, well-matched choice; 3 for an adequate but under-motivated choice; 0 for a mismatch or trivial single-call wrapper |
| Effectiveness of the core pipeline (e.g. retrieval quality, agent planning, prompt robustness) | 5 marks for a demonstrably effective pipeline; 3 for partial effectiveness; 0 if the pipeline does not function |
| Handling of failure modes (hallucination, tool errors, malformed output, empty retrieval) | 5 marks for graceful, tested handling; 3 for some handling; 0 for none |
| Reproducibility and engineering quality (runs cleanly from the entry point, config-driven, logged) | 5 marks for excellent reproducibility; 3 for some friction; 0 for non-reproducible |
| Efficiency and appropriateness of resource use (cost, latency, context length choices) | 5 marks for well-reasoned, efficient choices; 3 for adequate; 0 for unjustified or wasteful use |

**Provision of specific evaluation scenarios [10 marks]:** you must provide **five test scenarios** (e.g. query sets, perturbations, or edge cases) that exercise your solution, specified in your `config.yaml`/evaluation script and described in the report (see Section 2.2, Evaluation Scenarios). You will be assessed on how effectively your scenarios probe different aspects of your solution (at most 2 marks per scenario; scenarios that exercise materially different aspects — e.g. adversarial input, out-of-domain queries, long-context stress, ambiguous requests — score more highly than near-duplicates of one another).

**Provided materials:** you will be given a small starter repository containing a shared evaluation-logging utility and, where relevant to your chosen domain, a sample dataset. You are not allowed to modify the logging schema it defines, as this is used to standardise reported metrics across the cohort. Third-party packages are permitted, provided they are cited in your README and pinned in your dependency file.

Note that a solution which cannot be run from a clean environment following your README may be awarded 0 implementation marks.

### 2.2 Report (55%)

You must write a report that details the design, implementation, and evaluation of your solution, and reflects on its safety and ethical implications. The report must be structured as follows.

**Overview [5 marks]:** describe at a high level the problem you chose and the methodology you used. Explain what influenced your choice of technique(s) [200 words]. Marks awarded for a concise and precise description.

**Architecture [12 marks]:** include a diagram of your system's architecture and a description referencing it. Identify the components — e.g. retriever, vector store, orchestrator, agent loop, tool interfaces, guardrails — responsible for each part of the pipeline [300 words].

| Criterion | Marks |
|---|---|
| Appropriateness of the diagram | 3 |
| Depth, breadth and clarity of the discussion of the diagram | 4 |
| Correct identification and description of components | 5 |

**Technique Deep-Dive [13 marks]:** explain in technical depth *how* your chosen technique (RAG, fine-tuning, agentic tool use, or structured prompting) works in your system — e.g. chunking and embedding strategy and why you chose it, or your training/adaptation setup and hyperparameters, or your agent's planning and tool-selection loop, or your prompt structure and output-validation strategy. You should consider failure modes specific to the technique (e.g. retrieval mismatch, catastrophic forgetting, tool-call loops, prompt injection) and how your design mitigates them [350 words].

| Criterion | Marks |
|---|---|
| Appropriateness of the technique for the task | 3 |
| Clarity of the technical description | 3 |
| Demonstrated understanding of the technique's failure modes | 4 |
| Justification of design decisions | 3 |

**Evaluation [12 marks]:** evaluate your solution using one or more methodologies covered in the course (e.g. automatic metrics, LLM-as-judge, human evaluation, ablation, cost/latency analysis) [600 words].

At most 3 marks per chosen evaluation method [9 marks]; justification of the chosen evaluation methods [3 marks].

**Safety [5 marks]:** identify safety hazards specific to your application and mitigation measures appropriate were it to be deployed in a real-world setting. Include a table detailing hazards, guiding words, deviations, possible causes, consequences, and mitigating actions [300 words].

| Number | Guiding Words | Deviation | Possible Causes | Consequences | Mitigating Actions |
|---|---|---|---|---|---|
| Hazard 1 | | | | | |
| … | | | | | |
| Hazard N | | | | | |

*Table 1. Example table for hazard identification.*

| Criterion | Marks |
|---|---|
| Conciseness of the table's contents | 1 |
| Relevance of the hazards to the designed solution | 2 |
| Appropriateness of the hazard analysis | 2 |

**Ethics [5 marks]:** discuss the ethical implications of your chosen problem and your specific solution — e.g. bias, consent, data provenance, misuse potential, environmental cost, or displacement of human judgement. Reflect on how these relate to your design choices [300 words].

| Criterion | Marks |
|---|---|
| Identification of ethical implications | 1 |
| Depth, breadth and clarity of the discussion | 2 |
| Demonstrated understanding of implications for real-world deployment | 2 |

**Evaluation Scenarios [3 marks]:** include a table summarising, for each of your five scenarios, the configuration used (e.g. retriever settings, model, temperature, agent step limit) and its purpose, and explain how the selection of scenarios contributes to validating your solution [300 words, not including the table].

| | Model | Technique config | Scenario purpose | … |
|---|---|---|---|---|
| Scenario 1 | | | | |
| … | | | | |
| Scenario 5 | | | | |

*Table 2. Example table for specifying evaluation scenarios.*

The remaining **[5 marks]** of the report mark are based on presentation (structure, figures, adherence to structure, and referencing).

Your report should be formatted using A4 paper size, with a minimum font size of 11pt and minimum margins of 2cm, and references should follow the IEEE referencing style. The report must not exceed the word limit of 2500 words (excluding references). If your report exceeds this limit, the marker will stop reading at the limit and base the mark on what they have read so far. There is no limit on the number of images.

---

## 3. Submission

### 3.1 Anonymity

Your examination number, name, username, email address, or any other identifying information must not be present anywhere in your submission — including code comments, commit history, environment variables, terminal output shown in figures, and PDF metadata.

### 3.2 Electronic Submission

Submit your deliverables via the VLE submission point as two separate files: your report as a **PDF**, and your implementation as a single **ZIP** file. The root of your ZIP file should be your project repository, structured as shown below.

```
submission.zip
├── README.md
├── requirements.txt (or environment.yml)
├── config.yaml
├── run.py
├── evaluation/
│   └── (evaluation harness and scenario configs)
├── src/
│   └── (your application code)
└── (any other supporting files)
```

You should exclude virtual environments, model checkpoints above 50MB, `.git` folders (these may contain identifying information), and any API keys or `.env` files. Your README must state clearly how a marker can obtain or substitute their own API key to run your solution.

It is your responsibility to ensure your implementation runs from a clean environment following your own README before submission.

**END OF PAPER**
