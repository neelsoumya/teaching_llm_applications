# Guest Lectures

Guest lectures bring practitioner and research perspectives that complement the core weekly material. Guests speak for approximately 45 minutes followed by 15 minutes of Q&A. Students are encouraged to read any suggested background material before each session.

---

## Speakers

### Amit More

- Energy 
- Tools
- Industry whitepapers and industry specific problems
- [Energy](energy.md)

## Julio Santiseteban

- Peru
- Quechua (indigenous language)
- Oral traditions (Amazon) oral/speech models
- GPT-Latam

### Nirav Bhatt

- Low resource languages Swahili
- Practicals on creating SLMs/LLMs on low resource languages

## Ignatius Ezeani

- low resource languages
- Nigeria
- LLMs for low resource languages
- 2000 languages in Africa; none of them represented well enough
- challenges in building LLMs/SLMs in low resource contexts
- [SLMs](BERT_SLMs.md)

### Rashmita Soni

- LLMs in industry

### Cole Robertson — Speech, Language, and LLMs in Industry

**Background.** Cole Robertson is the co-founder and CTO of a speech-AI startup building real-time spoken language understanding systems on top of large language models. His work sits at the intersection of automatic speech recognition (ASR), natural language understanding (NLU), and production LLM deployment.

**Talk overview.** Cole will discuss the engineering and product challenges of combining speech models with LLMs at scale. Topics will include:

- How ASR outputs (disfluencies, punctuation-free transcripts, speaker diarisation artefacts) interact with LLM tokenisation and context windows.
- Latency constraints in real-time spoken dialogue systems and strategies for streaming LLM inference.
- The startup experience: choosing open-source vs proprietary models, managing inference costs, and iterating on prompts under production pressure.
- Lessons learned from deploying to enterprise customers: reliability, hallucination in spoken contexts, and user trust.

**Relevant course connections.** Weeks 1 (LLM overview), 8 (prompting), 12 (production deployment).

**Suggested background reading.**
- [Whisper paper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356) — OpenAI's ASR model, commonly used upstream of LLMs in speech pipelines.
- [Spoken Dialogue Systems survey](https://arxiv.org/abs/2306.10349)

---

### Glasgow Startup — LLMs for Real-World Applications

**Background.** This talk features the founding team of a Glasgow-based AI startup applying LLMs to a domain-specific business problem. Details to be confirmed; the talk will focus on the product development lifecycle from prototype to deployment.

**Talk overview.** The founders will walk through their experience building an LLM-powered product from scratch, covering:

- Problem selection: how they identified a domain where LLMs provided genuine value over existing solutions.
- Prototyping rapidly with off-the-shelf models and prompting before committing to fine-tuning.
- The decision of when (and whether) to fine-tune vs use RAG vs rely on prompt engineering.
- Evaluation in the wild: how they measure whether the product is actually working for users.
- Commercial and ethical considerations: data privacy, customer trust, and responsible AI in a startup context.

**Relevant course connections.** Weeks 7 (fine-tuning), 8 (prompting), 9 (RAG), 11 (evaluation), 12 (applications).

**Suggested background reading.**
- [LLM-powered applications survey (Zhao et al., 2023)](https://arxiv.org/abs/2303.18223)
- Course notes: [week12_applications_and_project.md](week12_applications_and_project.md)

---

## How to Prepare

Before each guest lecture, students should:

1. Review the suggested background reading (listed above for each speaker).
2. Look up the speaker's prior work or company — understanding their context makes the Q&A more productive.
3. Prepare at least one question in advance. Good question categories:
   - *Technical depth*: "How do you handle X in practice?" or "What did you try that didn't work?"
   - *Trade-offs*: "Why did you choose X over Y?"
   - *Failure modes*: "What surprised you most when you deployed?"
   - *Career/research path*: "How did you move from research to product?" or "What skills do you wish you had earlier?"

---

## Potential Future Speakers

The following areas are under consideration for future guest lectures. If you have a contact who works in one of these areas and would be willing to speak, please let the course organiser know.

| Area | Topics of interest |
|------|--------------------|
| Healthcare AI | Clinical NLP deployment, regulatory compliance, PPIE |
| AI Safety | Red-teaming, alignment research, responsible scaling |
| Foundation model research | Pre-training at scale, data curation, evaluation |
| Legal / policy | AI regulation (EU AI Act, UK frameworks), IP and copyright |
| Open-source LLMs | Community model development, GGUF/llama.cpp, Ollama |
| Multimodal systems | Vision-language models, speech + vision |

---

## Notes for Speakers

If you are a guest speaker, thank you for contributing to this course. Practical tips:

- **Audience**: MSc students in Computer Science with 12–13 weeks of LLM background. They have implemented attention, trained small transformers, built RAG pipelines, and written agents. You do not need to explain basic concepts, but please do not assume familiarity with your specific domain.
- **Format**: 45-minute talk + 15-minute Q&A. Slides are welcome but not required.
- **Code**: live demos or code walkthroughs are very well received.
- **Candour**: students benefit most from honest accounts of what went wrong, what trade-offs you made, and what you would do differently. Polished success stories are less useful than genuine engineering experience.
- **Contact**: soumya.banerjee@york.ac.uk / sb2333@cam.ac.uk
