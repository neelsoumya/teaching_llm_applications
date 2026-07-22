# Week 9 — Retrieval-Augmented Generation (RAG)

## Lecture Overview

- [Code from Alammar book](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models/tree/main/chapter08)

- LLMs have a knowledge cutoff and limited context windows. They also hallucinate. **Retrieval-Augmented Generation (RAG)** addresses these limitations by grounding the model in retrieved documents at inference time — no retraining required.

---

## 1. The Problem RAG Solves

| Problem | RAG Solution |
|---------|-------------|
| Knowledge cutoff | Retrieve from an up-to-date document store |
| Hallucination | Ground answers in retrieved evidence |
| Context window limits | Only inject the most relevant chunks |
| Source attribution | Citations come from retrieved documents |
| Domain-specific knowledge | Index domain documents once; use always |

---

## 2. RAG Architecture

A RAG pipeline has two phases:

### Phase 1 — Indexing (offline, done once)

```
Documents
    │
    ▼
Chunking
    │
    ▼
Embedding (each chunk → dense vector)
    │
    ▼
Vector Store (index of chunk vectors)
```

### Phase 2 — Retrieval and Generation (online, per query)

```
User query
    │
    ▼
Embed query
    │
    ▼
Similarity search in vector store → top-k chunks
    │
    ▼
Build prompt: system + retrieved chunks + query
    │
    ▼
LLM generates answer grounded in retrieved context
```

---

## 3. Chunking Strategies

How you split documents matters enormously.

| Strategy | Description | When to use |
|----------|-------------|-------------|
| Fixed-size | Split every N tokens, with M-token overlap | Simple baseline |
| Sentence | Split on sentence boundaries | Preserves semantic units |
| Recursive character | Try paragraph, then sentence, then character | Good general default |
| Semantic | Group sentences by embedding similarity | Best quality; slower |
| Document structure | Split on headers, sections | Structured docs (PDFs, HTML) |

**Overlap** between chunks (e.g. 50–100 tokens) ensures that information spanning a chunk boundary is not lost.

---

## 4. Embedding Models for Retrieval

Not all embedding models are equal for retrieval. Key benchmarks: MTEB (Massive Text Embedding Benchmark).

| Model | Embedding dim | Notes |
|-------|--------------|-------|
| `all-MiniLM-L6-v2` | 384 | Fast, small, good baseline |
| `all-mpnet-base-v2` | 768 | Better quality |
| `text-embedding-3-small` (OpenAI) | 1536 | Strong, API-based |
| `nomic-embed-text-v1.5` | 768 | Open-source, competitive |
| `bge-large-en-v1.5` | 1024 | Strong open-source |

Use **asymmetric** retrieval: query and document can be embedded differently (bi-encoder approach).

---

## 5. Vector Stores

A **vector store** stores embeddings and supports approximate nearest neighbour (ANN) search.

| Store | Type | Notes |
|-------|------|-------|
| FAISS | Library | Fast, in-memory, Facebook AI |
| ChromaDB | Embedded DB | Easy setup, good for dev |
| Pinecone | Managed cloud | Scalable, paid |
| Weaviate | Open source | Graph + vector |
| pgvector | PostgreSQL extension | Good if already using PG |
| Qdrant | Open source | Good performance |

For prototyping, FAISS or ChromaDB are standard.

```python
import faiss
import numpy as np

d = 384   # embedding dimension
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings, dtype='float32'))

# Search
D, I = index.search(np.array([query_embedding], dtype='float32'), k=5)
# I contains indices of top-5 most similar chunks
```

---

## 6. Retrieval Methods

### 6.1 Dense Retrieval (Semantic Search)

Embed query and documents; return top-k by cosine similarity. The standard RAG approach.

### 6.2 Sparse Retrieval (BM25)

Keyword-based TF-IDF style scoring. Still competitive for exact-match queries.

### 6.3 Hybrid Retrieval

Combine dense and sparse scores (e.g. with reciprocal rank fusion). Generally outperforms either alone.

### 6.4 Re-ranking

After retrieving top-k candidates, use a cross-encoder to re-rank them. Cross-encoders jointly attend to query and document — more accurate but slower.

```
Query + Document → CrossEncoder → scalar relevance score
```

---

## 7. Building a RAG Pipeline with LangChain

```python
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 1. Load documents
loader = DirectoryLoader("./docs/", glob="**/*.txt")
docs = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed and index
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedder)

# 4. Build QA chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 4}))

# 5. Query
answer = qa.invoke("What are the main findings of the study?")
print(answer["result"])
```

---

## 8. Advanced RAG Patterns

### 8.1 HyDE (Hypothetical Document Embeddings)

Ask the LLM to generate a hypothetical answer, then use *that* as the query embedding. Often improves retrieval quality.

### 8.2 Multi-query Retrieval

Generate multiple reformulations of the query; retrieve for each; merge and deduplicate.

### 8.3 Parent-Child Chunking

Index small chunks for precise retrieval, but return the larger parent chunk for context.

### 8.4 Self-RAG

The model decides *when* to retrieve — not every query needs retrieval (Asai et al., 2023).

### 8.5 Corrective RAG (CRAG)

Evaluate retrieved documents for relevance; trigger web search if retrieved documents are insufficient.

---

## 9. Evaluating RAG

| Metric | Measures |
|--------|---------|
| Retrieval recall | Are the relevant documents retrieved? |
| Context precision | Are retrieved chunks actually relevant? |
| Faithfulness | Does the answer stay within the retrieved context? |
| Answer relevance | Does the answer address the question? |

Frameworks: **RAGAS**, **TruLens**, **DeepEval**.

---

## 10. Practical This Week

See `practicals/week09_practical.py`:
- Build a RAG pipeline over a set of documents (e.g. Wikipedia articles or course PDFs).
- Implement chunking, embedding, FAISS indexing, and query answering.
- Compare retrieval quality: dense only vs hybrid (BM25 + dense).
- Evaluate faithfulness: does the answer contain claims not in the retrieved context?

---

## 11. Further Reading

- Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — https://arxiv.org/abs/2005.11401
- Gao et al. (2023) — "Retrieval-Augmented Generation for Large Language Models: A Survey" — https://arxiv.org/abs/2312.10997
- Asai et al. (2023) — "Self-RAG" — https://arxiv.org/abs/2310.11511
- [RAGAS evaluation framework](https://github.com/explodinggradients/ragas)
- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/)

---

## Discussion Questions

1. What are the failure modes of RAG? Give two scenarios where RAG might produce a worse answer than no retrieval.
2. Explain the difference between a bi-encoder and a cross-encoder for retrieval. What are the trade-offs?
3. Design a RAG system for answering questions about a hospital's clinical guidelines. What chunking strategy would you use, and why?
4. How would you evaluate whether your RAG system is faithful — i.e. not hallucinating beyond the retrieved context?
