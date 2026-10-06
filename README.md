# Agentic GraphRAG for Olympics Question Answering

A question-answering system over an Olympics corpus that compares three retrieval pipelines: plain RAG, GraphRAG, and an Agentic GraphRAG that plans, retrieves in several steps and uses structured tools for counting and ranking questions.

## Highlights

- **Embeddings:** BGE
- **Vector store:** ChromaDB
- **LLM:** qwen2.5:7b-instruct served locally through Ollama
- **Graph:** TigerGraph (Chunk, Entity and MENTIONS) plus a local entity index
- **Techniques:** vector search, graph-based retrieval, re-ranking, context compression, structured reasoning for aggregation and temporal queries

## Results

Latest benchmark on 20 Olympics questions (`python rag_pipeline.py compare --limit 20`, saved to `benchmark.json`):

| Pipeline | Accuracy | Lenient (>=80% of gold words) | Tokens/question | Sec/question |
|---|---|---|---|---|
| RAG | 10.0% | 10.0% | 1534 | 13.4 |
| GraphRAG | 33.3% | 30.0% | 846 | 2.07 |
| Agentic GraphRAG | 83.3% | 80.0% | 641 | 1.96 |
| **Orchestrated** | **85.0%** | 85.0% | 933 | 2.6 |

The orchestrator routed 16 questions to the agentic pipeline, 3 to GraphRAG for single-hop questions, and 1 single-hop question to the agentic pipeline.

Agentic GraphRAG is both the most accurate single pipeline and the cheapest in tokens, because counting, ranking and venue-plus-date questions are answered from the structured fact table without a planner call. The RAG time includes first-run model loading, so compare its latency with care.

An earlier run on a different, larger question set (before the structured fact table was fixed and enabled) gave RAG 40%, GraphRAG 43.3% and Agentic GraphRAG 60%, with the agentic pipeline taking about 20 s per query.

### Architecture

The system is organized into two phases: **offline indexing** and **online query processing**.

#### Indexing Phase — Offline

```text
                         Olympics Corpus
                               │
                               ▼
                            Chunking
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
              BGE Embeddings       Entity Extraction
                    │                  (Regex)
                    ▼                     │
                ChromaDB                 ▼
              Vector Store         entity_index.json
                    │              (Entity → Chunk IDs)
                    │                     │
                    └──────────┬──────────┘
                               │
                               ▼
                          TigerGraph
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        Chunk ── MENTIONS ── Entity    Structured Facts
                                      (events, venues,
                                       dates, medals)
```

#### Query-Time Phase — Online

```text
                         User Question
                               │
                               ▼
                    ┌───────────────────┐
                    │    Orchestrator   │
                    │      (Router)     │
                    └─────────┬─────────┘
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
            RAG           GraphRAG       Agentic GraphRAG
       Similarity Search  Graph +       Planner + Dynamic
                         Content        Sub-questions
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │    Retrieval Layer      │
                 │                         │
                 │ • ChromaDB Search       │
                 │ • TigerGraph Expansion  │
                 │ • Re-ranking            │
                 │ • Context Compression   │
                 └────────────┬────────────┘
                              │
                              ▼
                       Qwen 2.5 (Ollama)
                              │
                              ▼
                           Answer
```

### Pipeline Comparison

| Pipeline | Main Strategy | Retrieval |
|---|---|---|
| **RAG** | Semantic similarity | ChromaDB |
| **GraphRAG** | Entity and relationship traversal | TigerGraph + supporting content |
| **Agentic GraphRAG** | Dynamic planning and tool selection | ChromaDB + TigerGraph + reasoning |

The **Agentic GraphRAG** pipeline dynamically decides which retrieval strategy to use based on the question, available evidence, and information still required to answer the query.

## The three pipelines

1. **RAG:** embeds the question, retrieves the top chunks from ChromaDB and asks qwen2.5:7b-instruct to answer from them.
2. **GraphRAG:** adds graph retrieval. Entities found in the question are used to pull related chunks through the entity graph. Results are merged with vector hits, re-ranked and compressed before generation.
3. **Agentic GraphRAG:** parses the question first. Counting, ranking and venue-plus-date questions are answered directly from the structured fact table (`facts.py`). Other questions are split by a planner into sub-questions, each answered with retrieval, and the collected facts are combined in a final answer step.

On top of these, the **orchestrator** classifies each question as single-hop or multi-part and routes it to GraphRAG or Agentic GraphRAG. Plain RAG is kept as a baseline for comparison.

## Project structure

```
AgenticGraphRag/
├── rag_pipeline.py     # chunking, embeddings, ChromaDB, RAG / GraphRAG, CLI (index, ask, eval)
├── agentic.py          # Agentic GraphRAG: planner, sub-question loop, structured table tool
├── facts.py            # parses infobox facts; count / rank / lookup by venue and date
├── tg_graphrag.py      # TigerGraph: check, schema, load
├── entity_index.json   # generated by `index`: entity -> chunk ids
├── eval_results_*.jsonl# per-question evaluation output
└── README.md
```

## Setup

Requirements: Python 3.12, [Ollama](https://ollama.com) with qwen2.5:7b-instruct pulled, and a TigerGraph Cloud instance (optional, only for the TigerGraph graph backend).

```powershell
# install dependencies in your environment (uv / pip), then:
ollama pull qwen2.5:7b-instruct
ollama list          # confirm Ollama is running
```

Configure your TigerGraph host and credentials where `tg_graphrag.py` expects them.

## Usage

**1. Build the vector index and entity index**

```powershell
python rag_pipeline.py index --rebuild
```

This chunks the corpus, embeds it into ChromaDB and writes `entity_index.json`. Without `--rebuild`, the command exits early if ChromaDB already contains chunks, and the entity index is not written.

**2. Load the graph into TigerGraph**

```powershell
python tg_graphrag.py check     # instance must be "Ready"
python tg_graphrag.py schema    # creates Chunk, Entity, MENTIONS, graph and loading jobs
python tg_graphrag.py load      # uploads chunks, entities and mentions
```

**3. Ask a question and evaluate**

```powershell
python rag_pipeline.py eval --limit 5      # quick check
python rag_pipeline.py eval                # full evaluation
```

Evaluation prints accuracy (strict and lenient, at least 80% of gold words), average tokens per question, routing counts and a failure breakdown. Per-question details are saved to `eval_results_<mode>.jsonl`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `UnboundLocalError: ... 'facts'` in `agentic_graphrag` | A local variable named `facts` shadowed the `facts` module. Rename the local list (for example `fact_lines`). |
| `FileNotFoundError: entity_index.json` on `load` | Run `python rag_pipeline.py index --rebuild` first. |
| TigerGraph `502 Bad Gateway` | The cloud instance is paused or waking up. Resume it in the portal, wait a minute, rerun `check`. |
| "already exists" or "name is used by another object" during `schema` | Harmless. The schema was created on an earlier run. |
| Eval shows 0% with `retrieval_fail` and about 2 s total | The pipeline raised an exception, so the LLM was never called. Read the traceback above the summary. |

## Notes and limitations

- Most remaining errors are generation failures (the right context reached the LLM but the answer was wrong), so answer-extraction prompts are the next thing to improve.
- Questions the structured fact table cannot parse fall back to the planner, which is slower and costs more tokens.
- Entity extraction is regex-based, so entity quality depends on capitalization patterns in the corpus.
- Results come from only 20 questions, so one question is worth 5 percentage points. Run the full evaluation before drawing firm conclusions.
