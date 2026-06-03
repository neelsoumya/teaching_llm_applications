# Week 10 — Agents and Tool Use

## Lecture Overview

LLMs that only generate text are limited to what they know and can reason about in a single forward pass. **LLM agents** break these limitations by giving the model tools — the ability to search the web, execute code, call APIs, read files, and interact with the world. This week covers agent architectures, tool calling, the ReAct framework, and multi-agent systems.

---

## 1. What Is an LLM Agent?

An **agent** is an LLM that can:
1. **Observe** the environment (receive inputs: text, tool results, observations).
2. **Reason** about what to do next.
3. **Act** by calling tools or producing outputs.
4. **Iterate** — repeat the observe-reason-act loop until the task is complete.

The LLM is the "brain"; tools extend its capabilities beyond text generation.

---

## 2. The ReAct Framework

Yao et al. (2022) propose **ReAct** (Reasoning + Acting): interleave reasoning traces (Thought) and actions (Act/Observe) in the model's output.

```
Thought: I need to find the current population of York, England.
Action: search("population of York England 2024")
Observation: York has a population of approximately 210,000 as of 2024.

Thought: I now have the information. I can answer the question.
Final Answer: The population of York, England is approximately 210,000.
```

This makes the agent's reasoning transparent and debuggable.

---

## 3. Tool Calling (Function Calling)

Modern LLM APIs support **structured tool calling**: the model can emit a structured JSON call to a named function, the host executes it, and the result is fed back into the context.

### Defining Tools (OpenAI API format)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute a Python code snippet and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."}
                },
                "required": ["code"]
            }
        }
    }
]
```

### Agent Loop

```python
import json
from openai import OpenAI

client = OpenAI()
messages = [{"role": "user", "content": "What is the square root of the population of York?"}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto"
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            result = dispatch_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
    else:
        print(msg.content)
        break
```

---

## 4. Common Agent Tools

| Tool | Purpose |
|------|---------|
| Web search | Retrieve current information |
| Code execution | Run Python; do maths, data analysis |
| File I/O | Read/write files |
| Database query | Structured data retrieval (SQL) |
| REST API calls | External services (weather, calendars, etc.) |
| Email/calendar | Productivity automation |
| Browser use | Web scraping, form filling |
| RAG retrieval | Private document knowledge base |
| Image generation | DALL-E, Stable Diffusion |

---

## 5. Planning and Memory

### 5.1 Planning

For complex tasks the agent must plan before acting:
- **Zero-shot planning**: ask the LLM to produce a plan.
- **Tree of Thoughts (ToT)**: explore multiple reasoning paths in a tree structure; select the best.
- **Graph of Thoughts (GoT)**: more general than tree; allows merging reasoning paths.

### 5.2 Agent Memory

| Memory Type | Implementation |
|-------------|----------------|
| In-context (short-term) | The conversation history in the context window |
| External (long-term) | Vector store; retrieve relevant memories as needed |
| Episodic | Log of past actions and results |
| Procedural | Fine-tuned model weights or system prompt instructions |

---

## 6. Multi-Agent Systems

A single agent can be augmented by coordinating multiple specialised agents.

### 6.1 Orchestrator-Subagent Pattern

An orchestrator agent decomposes tasks and delegates to specialist subagents:

```
Orchestrator
    ├── Research Agent (web search, summarisation)
    ├── Code Agent (writes and executes code)
    └── Writer Agent (formats and polishes output)
```

### 6.2 Frameworks

- **LangGraph**: graph-based multi-agent orchestration.
- **AutoGen (AG2)**: conversational multi-agent framework; agents debate and collaborate.
- **CrewAI**: role-based agents with shared goals.
- **smolagents**: lightweight, minimal-dependency agent framework (HuggingFace).

```python
# Example using smolagents
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

result = agent.run("Find the 5 most cited papers on RAG published in 2023.")
print(result)
```

---

## 7. Agent Evaluation and Reliability

Agents are harder to evaluate than single-turn models because:
- Tasks span multiple steps; errors accumulate.
- Tool calls may fail or return unexpected results.
- The agent may loop, hallucinate tool calls, or give up prematurely.

### Key Evaluation Dimensions

| Dimension | What to measure |
|-----------|----------------|
| Task success rate | Does the agent complete the task correctly? |
| Efficiency | How many steps / tool calls does it take? |
| Faithfulness | Does it misrepresent tool results? |
| Robustness | Does it recover from tool failures? |
| Safety | Does it take unintended side-effecting actions? |

---

## 8. Safety Considerations for Agents

Agents that take real-world actions introduce new risks:
- **Irreversible actions**: sending emails, deleting files, making purchases.
- **Prompt injection via tool results**: retrieved web pages may contain adversarial instructions.
- **Scope creep**: an agent told to "book a flight" may also book a hotel without being asked.

Best practices:
- **Minimal tool set**: only provide tools needed for the task.
- **Human-in-the-loop**: require confirmation for high-stakes actions.
- **Sandboxed execution**: run code in an isolated environment.
- **Audit logs**: record all tool calls and results.

---

## 9. Practical This Week

See `practicals/week10_practical.py`:
- Build a ReAct-style agent from scratch using the OpenAI or Anthropic API with two tools: web search (simulated) and a Python code executor.
- Implement the agent loop and test on three multi-step tasks.
- Extend to a two-agent system (orchestrator + specialist) using a simple message-passing protocol.
- Evaluate the agent on 10 tasks and report success rate and average steps.

---

## 10. Further Reading

- Yao et al. (2022) — "ReAct: Synergizing Reasoning and Acting in Language Models" — https://arxiv.org/abs/2210.03629
- Wei et al. (2022) — "Chain-of-Thought Prompting" — https://arxiv.org/abs/2201.11903
- Wang et al. (2023) — "A Survey on Large Language Model based Autonomous Agents" — https://arxiv.org/abs/2308.11432
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [AutoGen / AG2](https://ag2.ai/)
- Existing notebooks: `simple_agent_ag2.ipynb`, `deep_research_agent_ag2.ipynb`, `smolagents_websearch.ipynb`, `agentic_workflow_llm_opensource.ipynb`

---

## Discussion Questions

1. Explain the ReAct framework. Why is interleaving thought and action useful?
2. What is "prompt injection via tool results"? Give a concrete example and a mitigation.
3. Design a multi-agent system for automated literature review in a scientific domain. Specify the roles of each agent and the tools they need.
4. Why are agents harder to evaluate than single-turn question-answering models?
