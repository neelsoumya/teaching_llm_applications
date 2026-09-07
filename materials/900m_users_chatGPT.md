# 🤔 ❓ How does ChatGPT/OpenAI serve 900 million users? 


- Database cache/read only cache
- Schedulers
- Intelligent use of GPUs
- KV cache



![image](../images/900m_users_chatgpt.jpeg)

- also see [GPU chapter](GPUs.md)
- see [resource accounting](resource_accounting.md)
- see [Quantization](quantization.md)
- see [Architecture](architectures.md)

## Performance Engineering in Agentic LLM Applications

> *Case Study:* Claude Code & Bun Runtime Optimizations (Jared Sumner, Dylan Conway, David Wolff, Sosuke Suzuki)

---


Building agentic AI systems like **Claude Code** (an AI coding agent operating in the terminal) requires optimizing both **model execution** and the **application infrastructure** wrapping the model. While LLM response time (Time To First Token and generation throughput) is a major factor, significant latency and UX lag originate in non-model software layers:

1. **Tool Call Execution Loops** (~300ms reduction)
2. **Terminal Rendering & ANSI Text Wrapping** (33×–88× faster rendering via native `Bun.wrapAnsi`)
3. **CLI Startup / Runtime Cold-Starts** (-10ms reduction)

This document breaks down the system-level engineering behind these three optimizations, providing students with a deep dive into how performance engineering directly impacts AI developer tools.

---

## 1. Tool Call Optimization (~300ms Faster)

### The Problem: Agent Execution Loops
An agentic LLM does not merely stream response text to a screen. It operates in an **interactive loop**:

```
[User Input] ➔ [LLM Generation] ➔ [Tool Selection & Arg Output] ➔ [Permission/Validation] ➔ [Execution] ➔ [Result Feed back to LLM]
```

When an agent needs to edit a file, run `ripgrep`, or execute a bash command, every millisecond spent outside model generation adds directly to the end-to-end user latency. A ~300ms latency reduction per tool call significantly increases responsiveness, especially for multi-turn agent tasks requiring 10+ sequential tool invocations.

```
+-----------------------------------------------------------------------------------+
| Naive Tool Call Flow (~300–400ms overhead)                                        |
| [LLM Complete JSON] ➔ [Parse JSON] ➔ [Shell Spawning: sh -c] ➔ [Exec] ➔ [Capture] |
+-----------------------------------------------------------------------------------+
| Optimized Tool Call Flow (~10–30ms overhead)                                      |
| [Stream JSON] ➔ [Early Arg Parse] ➔ [Direct System Call / Warmed Subshell]       |
+-----------------------------------------------------------------------------------+
```

### Architectural Techniques for ~300ms Speedups

#### A. Eliminating Shell Spawning Overhead
* **Naive Approach:** Executing system commands via high-level wrappers like `child_process.exec('sh -c "git status"')` or spawning isolated shell sessions. Launching a new shell process incurs shell initialization scripts, environment parsing, and process overhead (~50ms–150ms per invocation).
* **Optimized Approach:**
  * Executing binaries directly via system calls (`execve`) without wrapper shell processes (`sh -c`).
  * Utilizing persistent, warmed background subshells or worker process pools over low-overhead IPC channels (unix domain sockets / native stdio) rather than launching new processes on every step.

#### B. Streamed Tool Argument Parsing & Early Execution
* **Naive Approach:** Waiting for the model to finish generating the entire JSON payload, closing the code block or tool signature, and then calling `JSON.parse()`.
* **Optimized Approach:**
  * Streaming JSON parsing: As tokens arrive from the API, the runtime incrementally parses the JSON stream.
  * As soon as mandatory parameters (e.g., target file path) are received and validated, preliminary operations (like reading file headers or checking cache status) can begin before the LLM finishes generating optional arguments.

#### C. Asynchronous Permission & Hook Pipelines
* **Naive Approach:** Synchronous blocking hooks that check safety policies, path sanitization, and filesystem states sequentially prior to tool execution.
* **Optimized Approach:** Parallelizing pre-flight validation checks, caching safety/authorization decisions for the session, and pipelining process execution so that system resources are allocated immediately.

---

## 2. Terminal UI (TUI) Rendering Optimization: `Bun.wrapAnsi`

### The Problem: Frame Budgets in Terminal UIs
Terminal User Interfaces (TUIs) like Claude Code function similarly to lightweight game engines. To maintain smooth 60 FPS scrolling and real-time streaming displays:

$$	ext{Frame Budget} = rac{1000	ext{ ms}}{60	ext{ frames}} pprox 16.67	ext{ ms per frame}$$

When streaming thousands of tokens into a terminal, the UI must constantly re-calculate layout, wrap text to match the current terminal column width, and redraw the screen.

### The Text-Wrapping Bottleneck
Wrapping text in a terminal is far harder than standard string wrapping because of two elements:
1. **ANSI Escape Codes:** Sequences like `[31m` (red text) or `[1m` (bold) do not take up visual horizontal space on the terminal screen, but exist in the string byte buffer. Naive character counting breaks visual formatting or splits ANSI codes across lines, corrupting the UI display.
2. **Unicode Width:** Characters are not all 1 byte wide. Emojis, full-width CJK characters, and combined grapheme clusters occupy variable byte lengths and visual widths (0, 1, or 2 terminal columns).

In standard JavaScript/Node.js toolchains, developers rely on npm packages like `wrap-ansi`. These libraries rely on heavy Regular Expressions (Regex) and JavaScript string splitting. 

* **Performance Impact:** On long conversation logs (8,000+ characters), pure JS text wrapping can consume **7ms to 10ms per frame**, consuming over 50% of the entire 16.67ms frame budget and causing visible micro-stuttering, input lag, and screen tearing.

### The Solution: Native Engine Implementation (`Bun.wrapAnsi`)
By moving `wrapAnsi` directly into the runtime layer (C++/Zig inside Bun), the operation transforms from an interpreted JavaScript loop into a fast, native single-pass operation.


#### How `Bun.wrapAnsi` Works Under the Hood
1. **Single-Pass State Machine:** Instead of running multiple Regex passes to find ANSI codes, `Bun.wrapAnsi` uses a native state machine that scans bytes sequentially in a single pass.
2. **ANSI Code Tracking:** It maintains an active formatting state (color, background, font style) as it traverses byte offsets without splitting ANSI escape sequences across line breaks.
3. **Column-Aware Width Calculation:** It computes Unicode display widths natively in compiled C++/Zig code, placing hard line breaks only at visual boundary limits.
4. **Direct Buffer Slicing:** It avoids constructing intermediate JS string objects, returning memory-aligned string slices directly to the JS engine heap.

---

## 3. CLI Cold-Start Latency (-10ms Reduction)

### The Problem: Perceived Latency for Developers
For command-line tools, startup latency shapes user perception. Command-line interactions expect sub-100ms startup times. Any startup latency over 200ms feels sluggish to developers accustomed to native binary tools like `git` or `ripgrep`.

### Techniques for Sub-10ms Reductions
Jared Sumner (creator of Bun and engineer at Anthropic) contributed optimizations to lower runtime cold-start overhead:

1. **JavaScriptCore (JSC) Heap Snapshots & Bytecode Caching:**
   * Instead of parsing, compiling, and evaluating the JS bootstrap code on every CLI execution, the engine serializes the pre-initialized VM heap state into disk cache.
   * On startup, the VM memory maps (`mmap`) this snapshot directly into memory, jumping straight to user code execution.
2. **Lazy Module Loading & Dynamic Imports:**
   * Heavy dependencies (e.g., full Markdown parsers, complex API wrappers) are deferred until explicitly required by user commands, reducing initial bundle evaluation time.
3. **Single-Binary Layout Optimization:**
   * Statically linking application code and native assets into a unified binary file, eliminating dynamic library resolution (`dlopen`) overhead at startup.

---

## 4. Key Takeaways for AI Performance Engineers

When teaching performance engineering for LLM applications, emphasize that **AI application latency is a multi-layer stack**:

$$	ext{Total Latency} = T_{	ext{TTFT}} + T_{	ext{TokenGen}} + \sum_{i=1}^{N} \left( T_{	ext{ToolOverhead}} + T_{	ext{ToolExec}} 
ight) + T_{	ext{UIRender}}$$

* **Model Latency ($T_{	ext{TTFT}}, T_{	ext{TokenGen}}$):** Handled by model provider hardware (GPUs/TPUs), speculative decoding, and model architecture choice.
* **Orchestration Latency ($T_{	ext{ToolOverhead}}$):** Optimized by software architecture (streamed JSON parsing, persistent processes, asynchronous permission pipelines).
* **Client & UI Latency ($T_{	ext{UIRender}}$):** Optimized by moving heavy string manipulation and formatting operations into native compiled runtime code (`Bun.wrapAnsi`).

---

### Discussion Questions for Students
1. Why does an agentic tool call overhead compound exponentially in multi-turn execution tasks compared to single-turn chats?
2. How do multi-byte Unicode characters and ANSI escape sequences complicate text wrapping in terminal user interfaces?
3. What are the trade-offs between spawning a new subshell for every tool invocation versus maintaining a persistent background worker process?
