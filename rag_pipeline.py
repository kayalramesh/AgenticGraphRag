"""
Olympics RAG + Orchestrator (token- and context-optimized)
==========================================================
Setup:
    pip install sentence-transformers chromadb ollama tqdm numpy
    # 100% open source stack: bge embeddings + ChromaDB + Qwen served locally by Ollama
    ollama pull qwen2.5:7b-instruct
    # The context window (CTX_WINDOW) is passed to Ollama on every request, no Modelfile needed.
 
Usage:
    python rag_pipeline.py index                      # steps 1-4 (load, chunk, embed, store)
    python rag_pipeline.py ask "Which city hosted the 2016 Summer Olympics?"
    python rag_pipeline.py eval                       # run eval_public.jsonl, print accuracy + token use
 
Pipelines (chosen by the orchestrator):
    P1  single_hop_rag   : retrieve -> pack -> answer                       (cheapest)
    P2  multi_hop_rag    : decompose -> retrieve per hop -> synthesize
    P3  graph_expanded   : entity-graph expansion (stand-in for TigerGraph GraphRAG)
    Router: single -> P1 ; multi -> P2 ; if a pipeline says INSUFFICIENT, escalate P1 -> P2 -> P3
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
 
import chromadb
import numpy as np
import ollama
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm
 
try:                                   # reads TG_HOST, TG_SECRET, ... from a local .env file if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv(override=True)           # values in .env win over stale terminal variables
except ImportError:
    pass
 
if __name__ == "__main__":
    sys.modules.setdefault("rag_pipeline", sys.modules[__name__])
 
 
# =====================================================================
# CONFIG
# =====================================================================
CORPUS_PATH = "data_clean.jsonl"
EVAL_PATH = "eval_public.jsonl"
DB_DIR = "./chroma_db"
ENTITY_INDEX_PATH = "./entity_index.json"
COLLECTION = "olympics"
 
EMBED_MODEL = "BAAI/bge-small-en-v1.5"          # small, fast, strong retrieval
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
 
OLLAMA_HOST = "http://localhost:11434"           # local Ollama server (fully open source, no API key)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct")   # any open model; set LLM_MODEL=llama3.1:8b in .env to switch
FIELD_UNITS = True          # split flattened infobox runs into key: value fields before compression (set False to compare)
LLM_NUM_GPU = None                               # None = use GPU if Ollama can; auto-falls back to CPU (0) after repeated CUDA crashes
 
# chunking (in words; ~1 word = 1.3 tokens)
CHUNK_WORDS = 180
OVERLAP_WORDS = 40
 
# context-window budgeting (tokens). Keep sum of prompt parts < CTX_WINDOW.
CTX_WINDOW = 3072
P1_CONTEXT_BUDGET = 1500      # single-hop evidence
HOP_CONTEXT_BUDGET = 860     # evidence per hop in multi-hop
P3_CONTEXT_BUDGET = 1400     # graph-expanded evidence
CHARS_PER_TOKEN = 2.8        # measured on this corpus (infobox/number-heavy text): budgets below are REAL LLM tokens
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # small open-source cross-encoder (~90 MB)
GRAPH_TOP_N = 3              # chunks kept after re-ranking in the GraphRAG pipeline
ANSWER_MAX_TOKENS = 60       # short answers = cheap + better for exact-match grading
HUB_CAP = 150                # ignore entities that appear in more chunks than this ("Olympic Games")
 
 
# =====================================================================
# STEP 1: RETRIEVE TEXT FROM THE JSONL DOCUMENTS
# =====================================================================
TEXT_KEYS = ("text", "content", "body", "article", "passage", "document")
ID_KEYS = ("id", "doc_id", "_id", "docid")
TITLE_KEYS = ("title", "name", "page")
 
 
def pick(row, keys, default=""):
    for k in keys:
        if k in row and row[k]:
            return row[k]
    return default
 
 
def clean(text):
    text = re.sub(r"\[\d+\]|\[citation needed\]", "", text)          # wiki citation marks
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\1.", text)            # == Heading == -> "Heading."
    return re.sub(r"\s+", " ", text).strip()
 
 
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
 
 
def load_corpus(path=CORPUS_PATH):
    docs = []
    for i, row in enumerate(read_jsonl(path)):
        text = pick(row, TEXT_KEYS)
        if isinstance(text, list):
            text = " ".join(map(str, text))
        docs.append({
            "doc_id": str(pick(row, ID_KEYS, i)),
            "title": str(pick(row, TITLE_KEYS, "")),
            "text": clean(str(text)),
        })
    print(f"[load] {len(docs)} documents")
    return docs
 
 
# =====================================================================
# STEP 2: OVERLAPPING, SENTENCE-AWARE CHUNKING
# =====================================================================
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"“(])')
 
 
def wc(s):
    return len(s.split())
 
 
def est_tokens(s):
    return int(len(s) / CHARS_PER_TOKEN) + 1
 
 
def chunk_doc(doc, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Pack whole sentences up to `size` words; carry the last ~`overlap` words into the next chunk."""
    sents = SENT_SPLIT.split(doc["text"])
    chunks, cur, cur_w, fresh = [], [], 0, 0
    for s in sents:
        w = wc(s)
        if cur and cur_w + w > size:
            chunks.append(" ".join(cur))
            carry, cw = [], 0
            for t in reversed(cur):                       # build the overlap from trailing sentences
                if cw + wc(t) > overlap:
                    break
                carry.insert(0, t)
                cw += wc(t)
            cur, cur_w, fresh = carry, cw, 0
        cur.append(s)
        cur_w += w
        fresh += 1
    if cur and fresh:
        chunks.append(" ".join(cur))
    return [{
        "id": f"{doc['doc_id']}::{i}",
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "chunk_idx": i,
        "text": c,
    } for i, c in enumerate(chunks)]
 
 
# =====================================================================
# STEP 3: EMBEDDINGS
# =====================================================================
_embedder = None
 
 
def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder
 
 
def embed_docs(texts):
    return embedder().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
 
 
def embed_query(q):
    return embedder().encode(QUERY_PREFIX + q, normalize_embeddings=True)
 
 
# --- lightweight entity extraction (used by router + graph pipeline) ---
STOP = {"The", "In", "On", "At", "He", "She", "It", "They", "This", "That", "These", "Those", "A", "An",
        "And", "Of", "For", "With", "By", "From", "As", "After", "Before", "During", "Which", "Who", "What",
        "When", "Where", "How", "Why", "Did", "Does", "Do", "Is", "Are", "Was", "Were", "His", "Her",
        "Their", "Its", "Also", "However", "Both", "Compare", "Name", "List", "Many", "Much"}
ENT_RE = re.compile(r"\b[A-Z][a-zA-Z'’\-]+(?:\s+(?:of|de|la|the)\s+[A-Z][a-zA-Z'’\-]+|\s+[A-Z][a-zA-Z'’\-]+)*")
 
 
def extract_entities(text):
    ents = set()
    for m in ENT_RE.finditer(text):
        parts = m.group(0).split()
        while parts and parts[0] in STOP:
            parts.pop(0)
        if parts:
            ents.add(" ".join(parts))
    return ents
 
 
# =====================================================================
# STEP 4: STORE CHUNKS + EMBEDDINGS IN A VECTOR DATABASE (ChromaDB)
#         (swap for TigerGraph Vector DB later - only build_index/retrieve change)
# =====================================================================
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
 
 
def build_index(rebuild=False):
    client = chromadb.PersistentClient(path=DB_DIR)
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() > 0:
        print(f"[index] collection already has {col.count()} chunks (use --rebuild to redo)")
        return col
 
    docs = load_corpus()
    chunks = [c for d in docs for c in chunk_doc(d)]
    print(f"[chunk] {len(chunks)} chunks (avg {np.mean([wc(c['text']) for c in chunks]):.0f} words)")
 
    entity_index = defaultdict(list)                       # entity -> chunk ids (poor-man's graph)
    for c in chunks:
        for e in extract_entities(c["title"] + ". " + c["text"]):
            entity_index[e].append(c["id"])
    Path(ENTITY_INDEX_PATH).write_text(json.dumps(entity_index), encoding="utf-8")
 
    B = 256
    for i in tqdm(range(0, len(chunks), B), desc="embed+store"):
        batch = chunks[i:i + B]
        # embed "Title: text" so the page title travels with every chunk, store the raw text
        vecs = embed_docs([f"{c['title']}: {c['text']}" for c in batch])
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=vecs.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[{"doc_id": c["doc_id"], "title": c["title"], "chunk_idx": c["chunk_idx"]} for c in batch],
        )
    print(f"[index] stored {col.count()} chunks in {DB_DIR}")
    return col
 
 
_entity_index = None
 
 
def entity_index():
    global _entity_index
    if _entity_index is None:
        _entity_index = json.loads(Path(ENTITY_INDEX_PATH).read_text(encoding="utf-8"))
    return _entity_index
 
 
# =====================================================================
# STEP 5-7: USER QUERY -> QUERY EMBEDDING -> RETRIEVER
# =====================================================================
@dataclass
class Hit:
    id: str
    doc_id: str
    title: str
    chunk_idx: int
    text: str
    score: float
 
 
_col = None
 
 
def collection():
    global _col
    if _col is None:
        _col = get_collection()
    return _col
 
 
def retrieve(query, k=6):
    qv = embed_query(query)

    candidate_k = max(100, k * 20)

    res = collection().query(
        query_embeddings=[qv.tolist()],
        n_results=candidate_k
    )

    hits = []

    for cid, doc, meta, dist in zip(
        res["ids"][0],
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0]
    ):
        hits.append(
            Hit(
                cid,
                meta["doc_id"],
                meta["title"],
                meta["chunk_idx"],
                doc,
                1 - dist
            )
        )

    return rerank(query, hits, k)
 
 
_reranker = None
 
 
def rerank(query, hits, top_n):
    """Cross-encoder re-ranking: the model reads query + chunk together and we keep only the best `top_n`.
    Fewer, better chunks = fewer prompt tokens AND less distraction for the LLM."""
    global _reranker
    if len(hits) <= top_n:
        return hits
    try:
        if _reranker is None:
            _reranker = CrossEncoder(RERANK_MODEL)
        scores = _reranker.predict([(query, f"{h.title}: {h.text}") for h in hits], show_progress_bar=False)
    except Exception as e:                      # offline / model missing: degrade gracefully
        log(f"reranker unavailable ({str(e)[:50]}) -> keeping retrieval order")
        return sorted(hits, key=lambda h: -h.score)[:top_n]
    ranked = sorted(zip(scores, hits), key=lambda p: -p[0])[:top_n]
    return [Hit(h.id, h.doc_id, h.title, h.chunk_idx, h.text, float(sc)) for sc, h in ranked]
 
 
# =====================================================================
# TOKEN / CONTEXT-WINDOW OPTIMIZATION: dedupe overlap -> budget -> compress
# =====================================================================
trace = []          # every context string built during the current ask(), used by evaluate()
cited = []          # page titles used as evidence by the pipeline that produced the answer (citations)
steps = []          # human-readable investigation path
current = {"route": None}   # route label of the question being answered (read by the agentic planner)
 
 
def log(msg):
    steps.append(msg)
 
 
def _norm_sent(s):
    return re.sub(r"\W+", " ", s.lower()).strip()
 
 
FIELD_SPLIT = re.compile(r"\s(?=[a-z][a-z_]{1,24}: )")
 
 
def units(text):
    """Sentences; long flattened infobox runs ('event: X games: Y venue: Z ...') are split into key: value fields so
    compression can keep the one field that answers the question instead of the whole run."""
    out = []
    for s in SENT_SPLIT.split(text):
        out.extend(FIELD_SPLIT.split(s) if FIELD_UNITS and wc(s) > 40 else [s])
    return out
 
 
def build_context(query, hits, budget):
    """
    1. Drop sentences already seen (removes the chunk-overlap duplicates and repeated facts).
    2. If the remainder still exceeds `budget`, keep only the sentences most similar to the query.
    3. Restore document order and group by title so the LLM reads coherent text.
    """
    items, seen = [], set()
    for h in hits:
        for pos, s in enumerate(units(h.text)):
            key = _norm_sent(s)
            if key and key not in seen:
                seen.add(key)
                items.append({"doc": h.doc_id, "chunk": h.chunk_idx, "pos": pos, "title": h.title, "s": s})
 
    if sum(est_tokens(i["s"]) for i in items) > budget:
        qv = embed_query(query)
        sims = embed_docs([f"{i['title']}: {i['s']}" for i in items]) @ qv      # the page title gives each field its context
        chosen, used = [], 0
        for idx in np.argsort(-sims):
            t = est_tokens(items[idx]["s"])
            if used + t > budget:
                continue
            chosen.append(idx)
            used += t
        items = [items[i] for i in chosen]
 
    items.sort(key=lambda i: (i["doc"], i["chunk"], i["pos"]))
    grouped = defaultdict(list)
    for i in items:
        grouped[i["title"]].append(i["s"])
    ctx = "\n".join(f"[{t}] {' '.join(ss)}" for t, ss in grouped.items())
    trace.append(ctx)
    cited.extend(t for t in grouped if t not in cited)
    return ctx
 
 
# =====================================================================
# STEP 8-9: PROMPT + RETRIEVED CONTEXT -> OPEN-SOURCE LLM
# =====================================================================
client = ollama.Client(host=OLLAMA_HOST)
meter = Counter()
 
SYSTEM = ("Answer using ONLY the context. Be concise: a short phrase or one sentence "
          "(list every item if the question asks for several). "
          "If the context does not contain the answer, reply exactly: INSUFFICIENT")
 
 
def llm(messages, max_tokens=ANSWER_MAX_TOKENS):
    global LLM_NUM_GPU
    for attempt in range(3):
        opts = {"temperature": 0, "num_predict": max_tokens, "num_ctx": CTX_WINDOW}   # context window set per request
        if LLM_NUM_GPU is not None:
            opts["num_gpu"] = LLM_NUM_GPU
        try:
            r = client.chat(model=LLM_MODEL, messages=messages, options=opts, keep_alive="30m")
            break
        except ollama.ResponseError as e:
            if attempt == 2:
                raise
            print(f"[llm] Ollama error, retrying: {str(e)[:90]}")
            if attempt == 1:                      # second failure in a row -> stop using the GPU
                LLM_NUM_GPU = 0
                print("[llm] falling back to CPU")
            time.sleep(3)
    meter["prompt"] += r.get("prompt_eval_count", 0) or 0
    meter["completion"] += r.get("eval_count", 0) or 0
    meter["calls"] += 1
    return r["message"]["content"].strip()
 
 
def answer_from_context(question, context, max_tokens=ANSWER_MAX_TOKENS):
    room = int((CTX_WINDOW - max_tokens - est_tokens(SYSTEM) - est_tokens(question) - 40) * CHARS_PER_TOKEN)
    if len(context) > room:                     # the prompt must always fit the context window
        context = context[:max(room, 200)]
        log("evidence trimmed to fit the context window")
    return llm([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
    ], max_tokens)
 
 
def is_insufficient(ans):
    return "INSUFFICIENT" in ans.upper()
 
 
# =====================================================================
# THE THREE PIPELINES
# =====================================================================
def single_hop_rag(q):
    """P1: one retrieval, one LLM call."""
    hits = retrieve(q, k=6)
    log(f"vector search -> {len(hits)} chunks")
    ctx = build_context(q, hits, P1_CONTEXT_BUDGET)
    return answer_from_context(q, ctx)
 
 
def decompose(q):
    out = llm([
        {"role": "system", "content": "Split the question into at most 3 short sequential sub-questions. "
                                      "Later ones may refer to earlier answers as #1, #2. "
                                      "Return ONLY a JSON list of strings."},
        {"role": "user", "content": q},
    ], max_tokens=120)
    m = re.search(r"\[.*\]", out, re.S)
    try:
        subs = json.loads(m.group(0)) if m else [q]
        return [str(s) for s in subs][:3] or [q]
    except json.JSONDecodeError:
        return [q]
 
 
def multi_hop_rag(q):
    """P2: decompose -> retrieve small context per hop -> synthesize from the hop answers only."""
    subs = decompose(q)
    log(f"decomposed into {len(subs)} sub-question(s)")
    if len(subs) == 1:
        return single_hop_rag(q)
    facts, answers = [], []
    for sq in subs:
        for j, a in enumerate(answers, 1):
            sq = sq.replace(f"#{j}", a)
        hits = retrieve(sq, k=4)
        ctx = build_context(sq, hits, HOP_CONTEXT_BUDGET)
        a = answer_from_context(sq, ctx, max_tokens=40)
        if is_insufficient(a):
            return "INSUFFICIENT"
        answers.append(a)
        facts.append(f"- {sq} => {a}")
    # final synthesis sees only the tiny fact list, not the raw chunks (big token saving)
    return answer_from_context(q, "\n".join(facts))
 
 
def graph_expand(q, seeds, n_extra=4):
    """Follow entity links from the question + seed chunks to pull in related chunks."""
    idx = entity_index()
    seed_ents = set()
    for h in seeds:
        seed_ents |= extract_entities(h.title + ". " + h.text)
    anchors = {e for e in (extract_entities(q) | seed_ents) if e in idx and len(idx[e]) <= HUB_CAP}
    votes = Counter()
    for e in anchors:
        for cid in idx[e]:
            votes[cid] += 1
    seed_ids = {h.id for h in seeds}
    cand = [cid for cid, _ in votes.most_common(40) if cid not in seed_ids]
    if not cand:
        return seeds
    got = collection().get(ids=cand, include=["embeddings", "documents", "metadatas"])
    sims = np.array(got["embeddings"]) @ embed_query(q)
    extra = []
    for i in np.argsort(-sims)[:n_extra]:
        m = got["metadatas"][i]
        extra.append(Hit(got["ids"][i], m["doc_id"], m["title"], m["chunk_idx"], got["documents"][i], float(sims[i])))
    return seeds + extra
 
 
def tg_enabled():
    import os
    return bool(os.environ.get("TG_HOST"))
 
 
def tg_hits(q):
    """Graph-aware retrieval. TigerGraph hybrid query when configured; if it is unreachable or errors,
    fall back to the local entity graph so the pipeline never fails because of the network."""
    if tg_enabled():
        try:
            from tg_graphrag import tg_retrieve
            hits = tg_retrieve(q, k_seed=6, k_extra=6)
            log(f"TigerGraph hybrid query -> {len(hits)} chunks")
            return hits
        except Exception as e:
            log(f"TigerGraph unavailable ({str(e)[:60]}) -> local entity-graph fallback")
    hits = graph_expand(q, retrieve(q, k=6), n_extra=6)
    log(f"local entity-graph expansion -> {len(hits)} chunks")
    return hits
 
 
# =====================================================================
# ORCHESTRATOR: route by single-hop vs multi-hop, escalate on failure
# =====================================================================
MULTI_CUES = re.compile(
    r"\b(both|compare|compared|difference|than|same|also|before|after|earlier|later|older|younger|"
    r"whose|that also|who also|how many .* and|combined|total of)\b", re.I)
 
 
def route(q):
    """Cheap heuristics first (0 tokens); only ambiguous questions cost one tiny LLM call."""
    try:
        from facts import parse as facts_parse
        if facts_parse(q):                                 # aggregation / structured lookup -> the agent's table tool
            return "multi"
    except ImportError:
        pass
    n_ent = len(extract_entities(q))
    cue = bool(MULTI_CUES.search(q))
    if cue and n_ent >= 2:
        return "multi"
    if not cue and n_ent <= 1 and wc(q) <= 14:
        return "single"
    out = llm([
        {"role": "system", "content": "Classify the question. 'single' = answerable from one fact/passage. "
                                      "'multi' = needs combining facts from 2+ passages, comparing, or chaining. "
                                      "Reply with one word: single or multi."},
        {"role": "user", "content": q},
    ], max_tokens=3)
    return "multi" if "multi" in out.lower() else "single"
 
 
def graphrag(q):
    """P2: seed chunks by vector search + entity hops + hybrid re-rank (one LLM call)."""
    hits = tg_hits(q)
    if not hits:
        return "INSUFFICIENT"
    hits = rerank(q, hits, GRAPH_TOP_N)
    log(f"re-ranked -> kept {len(hits)} chunks")
    return answer_from_context(q, build_context(q, hits, P3_CONTEXT_BUDGET), max_tokens=80)
 
 
def agentic(q):
    """P3: LLM-planned investigation with tool choice and reflection (see agentic.py)."""
    from agentic import agentic_graphrag
    return agentic_graphrag(q)
 
 
def best_effort(q, budget=1400):
    """Last resort when every pipeline said INSUFFICIENT: answer from the evidence already retrieved."""
    parts, used = [], 0
    for ctx in reversed(trace):
        t = est_tokens(ctx)
        if used + t <= budget:
            parts.append(ctx)
            used += t
    if not parts:
        return "INSUFFICIENT"
    evidence = "\n".join(reversed(parts))
    return llm([
        {"role": "system", "content": "Answer the question from the context as well as you can, in a few words. "
                                      "If unsure, give your best answer based on the context. Never reply INSUFFICIENT."},
        {"role": "user", "content": f"Context:\n{evidence}\n\nQuestion: {q}\nAnswer:"},
    ], ANSWER_MAX_TOKENS)
 
 
PIPELINES = {"rag": single_hop_rag, "graphrag": graphrag, "agentic": agentic, "multihop": multi_hop_rag}
 
# Orchestrator policy: which pipelines to try, in order, for each question type.
# Cheapest first; a pipeline answering INSUFFICIENT escalates to the next one.
CHAINS = {
    "single": ["rag", "graphrag", "agentic"],
    "multi": ["agentic", "graphrag"],
}
 
 
def ask(q, force=None):
    """force = 'rag' | 'graphrag' | 'agentic' | 'multihop' runs only that pipeline (for benchmarking each one)."""
    meter.clear()
    trace.clear()
    steps.clear()
    cited.clear()
    t0 = time.time()
    label = force or route(q)
    current["route"] = None if force else label
    log(f"router: {'forced ' if force else ''}{label}")
    ans, used = "INSUFFICIENT", None
    for name in ([force] if force else CHAINS[label]):
        cited.clear()                                   # citations belong to the pipeline that answered
        log(f"pipeline: {name}")
        ans, used = PIPELINES[name](q), name
        if not is_insufficient(ans):
            break
        log(f"{name}: not enough evidence -> escalate")
    if is_insufficient(ans) and not force:
        log("all pipelines exhausted -> best-effort answer from the evidence retrieved so far")
        ans, used = best_effort(q), "best_effort"
    return {
        "question": q, "route": label, "pipeline": used,
        "answer": ans if not is_insufficient(ans) else "Not found in corpus",
        "sources": list(cited)[:6], "path": list(steps),
        "prompt_tokens": meter["prompt"], "completion_tokens": meter["completion"],
        "llm_calls": meter["calls"], "seconds": round(time.time() - t0, 2),
    }
 
 
def safe_ask(q, force=None):
    try:
        return ask(q, force=force)
    except Exception as e:
        import traceback

        print("\n" + "=" * 80)
        print("QUESTION:", q)
        traceback.print_exc()
        print("=" * 80 + "\n")

        return {
            "question": q,
            "route": force or "error",
            "pipeline": "error",
            "answer": "Not found in corpus",
            "sources": [],
            "path": [f"error: {type(e).__name__}: {str(e)[:120]}"],
            "prompt_tokens": meter["prompt"],
            "completion_tokens": meter["completion"],
            "llm_calls": meter["calls"],
            "seconds": 0.0,
        }
 
 
def show(r):
    print(f"\nAnswer : {r['answer']}")
    print(f"Sources: {', '.join(r['sources']) or '-'}")
    print(f"Route  : {r['route']} -> {r['pipeline']}  |  tokens {r['prompt_tokens']}+{r['completion_tokens']}"
          f"  |  LLM calls {r['llm_calls']}  |  {r['seconds']}s")
    print("Investigation path:")
    for st in r["path"]:
        print("  -", st)
 
 
# =====================================================================
# EVALUATION on eval_public.jsonl
# =====================================================================
def _norm_ans(s):
    s = re.sub(r"\b(a|an|the)\b", " ", str(s).lower())
    return re.sub(r"\W+", " ", s).strip()
 
 
def correct(pred, gold):
    golds = gold if isinstance(gold, list) else [gold]
    p = _norm_ans(pred)
    return any(_norm_ans(g) and (_norm_ans(g) in p or (p and p in _norm_ans(g))) for g in golds)
 
 
def gold_in(contexts, gold):
    golds = gold if isinstance(gold, list) else [gold]

    blob = _norm_ans(" ".join(contexts))
    blob_words = set(blob.split())

    for g in golds:
        g = _norm_ans(g)

        if not g:
            continue

        # Exact match
        if g in blob:
            return True

        # Partial word overlap
        g_words = set(g.split())

        if not g_words:
            continue

        overlap = len(g_words & blob_words)

        if overlap >= max(1, int(len(g_words) * 0.5)):
            return True

    return False
 
 
def token_f1(pred, gold):
    golds = gold if isinstance(gold, list) else [gold]
    p = _norm_ans(pred).split()
    best = 0.0
    for g in golds:
        gt = _norm_ans(g).split()
        common = Counter(p) & Counter(gt)
        n = sum(common.values())
        if n and p and gt:
            pr, rc = n / len(p), n / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best
 
 
def token_recall(pred, gold):
    g = set(_norm_ans(gold).split())
    return len(g & set(_norm_ans(pred).split())) / len(g) if g else 0.0
 
 
def _collapse(s):
    return re.sub(r"\W+", "", str(s).lower())
 
 
def lenient_correct(pred, gold):
    """Strict containment OR >=80% of the gold answer's words present OR the same letters ignoring spaces/punctuation
    ('Dani KingLaura Trott' vs 'Dani King, Laura Trott')."""
    golds = gold if isinstance(gold, list) else [gold]
    return (correct(pred, gold) or any(token_recall(pred, g) >= 0.8 for g in golds)
            or any(_collapse(g) and _collapse(g) in _collapse(pred) for g in golds))
 
 
def analyze(name="agentic", n=10):
    """Print the wrong answers of an eval run, classified, so you can see WHY they fail."""
    path = f"eval_results_{name}.jsonl"
    rows = list(read_jsonl(path))
    bad = [r for r in rows if not r.get("correct")]
    kinds = Counter()
    lines = []
    for r in bad:
        pred = r.get("answer") or r.get("prediction") or r.get("predicted_answer") or r.get("pred") or ""
        gold = r.get("gold") or r.get("gold_answer") or r.get("expected_answer") or ""
        f1 = token_f1(pred, gold)
        kind = "abstained" if ("not found" in str(pred).lower() or is_insufficient(str(pred))) \
            else ("near-miss" if f1 >= 0.5 else "wrong")
        kinds[kind] += 1
        lines.append(f"- Q: {r.get('question')}\n    gold: {str(gold)[:100]}\n    pred: {str(pred)[:100]}   "
                     f"[{kind}, f1={f1:.2f}, gold_in_context={r.get('gold_in_context')}]")
    print(f"{len(bad)}/{len(rows)} wrong in {path}: {dict(kinds)}")
    print("(abstained = model said not found; near-miss = partly right/format; wrong = confident wrong answer)\n")
    print("\n".join(lines[:n]))
 
 
def evaluate(limit=None, force=None, out_path=None):
    """Scores answers AND tells you why a question failed: retrieval (gold never reached the LLM) or generation."""
    out_path = out_path or f"eval_results_{force or 'orchestrated'}.jsonl"
    rows = list(read_jsonl(EVAL_PATH))[:limit]
    ok = ok_len = tp = tc = 0
    secs = 0.0
    routes, diag = Counter(), Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"eval[{force or 'orchestrated'}]"):
            q = pick(row, ("question", "query", "q"))
            gold = pick(row, ("answer", "gold", "ground_truth", "expected_answer"))
            r = safe_ask(q, force=force)
            hit = gold_in(trace, gold)
            good = correct(r["answer"], gold)
            ok += good
            ok_len += lenient_correct(r["answer"], gold)
            tp += r["prompt_tokens"]
            tc += r["completion_tokens"]
            secs += r["seconds"]
            routes[f"{r['route']}->{r['pipeline']}"] += 1
            diag["correct" if good else ("generation_fail" if hit else "retrieval_fail")] += 1
            f.write(json.dumps({**r, "gold": gold, "correct": good, "gold_in_context": hit}, ensure_ascii=False) + "\n")
    n = max(len(rows), 1)
    summary = {"accuracy": ok / n, "accuracy_lenient": ok_len / n, "avg_prompt_tokens": tp / n, "avg_completion_tokens": tc / n,
               "avg_tokens": (tp + tc) / n, "avg_seconds": secs / n, "failures": dict(diag), "routing": dict(routes)}
    print(f"\nAccuracy: {ok}/{n} = {ok / n:.1%}   (lenient, >=80% of gold words: {ok_len / n:.1%})")
    print(f"Avg tokens/question: prompt {tp / n:.0f} + completion {tc / n:.0f} = {(tp + tc) / n:.0f}")
    print("Routing:", dict(routes))
    print("Failure breakdown:", dict(diag), "(retrieval_fail = answer text never reached the LLM)")
    print(f"Per-question details saved to {out_path}")
    return summary
 
 
def compare(limit=None):
    """Benchmark each pipeline on its own plus the orchestrator: accuracy vs token cost vs latency."""
    results = {}
    for mode in ("rag", "graphrag", "agentic", None):
        results[mode or "orchestrated"] = evaluate(limit, force=mode)
    print(f"\n{'pipeline':<13}{'accuracy':>10}{'lenient':>10}{'tokens/q':>12}{'sec/q':>9}")
    for name, sm in results.items():
        print(f"{name:<13}{sm['accuracy']:>10.1%}{sm['accuracy_lenient']:>10.1%}{sm['avg_tokens']:>12.0f}{sm['avg_seconds']:>9.1f}")
    Path("benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved benchmark.json")
    return results
 
 
def predict(input_path="eval_hidden.jsonl", output_path="predictions.jsonl", limit=None, resume=False):
    """Answer EVERY question in a file (public or hidden set) with the full orchestrator; no gold labels are used.
    A crash on one question is recorded and the run continues; --resume continues an interrupted run."""
    rows = list(read_jsonl(input_path))[:limit]
    done = set()
    if resume and Path(output_path).exists():
        done = {r.get("question") for r in read_jsonl(output_path)}
    with open(output_path, "a" if resume else "w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="predict"):
            q = pick(row, ("question", "query", "q"))
            if q in done:
                continue
            r = safe_ask(q)
            out = {("gold_answer" if k == "answer" else k): v for k, v in row.items()}
            out.update({"question": q, "answer": r["answer"], "sources": r["sources"], "pipeline": r["pipeline"],
                        "tokens": r["prompt_tokens"] + r["completion_tokens"], "path": r["path"]})
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
    print(f"Predictions for {input_path} are in {output_path}")
 
 
# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["index", "ask", "eval", "compare", "predict", "analyze"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", choices=sorted(PIPELINES), default=None)
    ap.add_argument("--input", default="eval_hidden.jsonl")
    ap.add_argument("--output", default="predictions.jsonl")
    ap.add_argument("--resume", action="store_true")
    a, _ = ap.parse_known_args()          # ignore extra args such as Jupyter's --f=kernel.json
 
    if a.cmd == "index":
        build_index(rebuild=a.rebuild)
    elif a.cmd == "ask":
        show(ask(a.question, force=a.force))
    elif a.cmd == "eval":
        evaluate(a.limit, force=a.force)
    elif a.cmd == "compare":
        compare(a.limit)
    elif a.cmd == "predict":
        predict(a.input, a.output, a.limit, a.resume)
    elif a.cmd == "analyze":
        analyze(a.force or "agentic", a.limit or 10)
    else:
        print("Commands: index | ask \"question\" | eval | compare | predict | analyze   (notebook: call ask(), evaluate(), compare(), predict())")
 