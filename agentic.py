"""
Agentic GraphRAG  (pipeline 3 of 3)
===================================
An LLM-planned investigation instead of a single retrieval:

    plan      the LLM splits the question into <= 3 sub-questions and picks a retrieval TOOL for each
    retrieve  tools: page   = look up a named Wikipedia page by title (no LLM tokens, exact)
                     graph  = TigerGraph hybrid graph+vector retrieval (entity hops), local fallback
                     vector = plain semantic search
    answer    each hop is answered from a small, de-duplicated evidence budget
    table     structured facts: count / rank / look up event pages by venue+date (see facts.py), zero planner tokens
    reflect   if a hop finds nothing, widen ONCE using all three tools before giving up
    finish    1 hop  -> its answer is the final answer (no extra LLM call)
              >1 hop -> one short synthesis call over the hop answers only

Why it is token-efficient: easy questions cost one planner call + one answer call; evidence is capped per hop,
and the final synthesis never re-reads raw chunks. Sources (page titles) are collected in rag_pipeline.cited and
the investigation path in rag_pipeline.steps, so every answer ships with citations and a trace.
"""
import json
import math
import re
from collections import Counter

import numpy as np

import facts
import rag_pipeline as rp

MAX_HOPS = 3
HOP_TOP_N, HOP_BUDGET = 2, 1000      # normal hop: best 2 chunks after re-ranking (real LLM tokens)
WIDE_TOP_N, WIDE_BUDGET = 4, 1800    # widened hop (only when the first try found nothing)

PLAN_SYSTEM = (
    "You plan how to answer a question over Olympics Wikipedia pages. Split it into at most 3 short sequential "
    "sub-questions (later ones may refer to earlier answers as #1, #2). Use as few as possible: one if one fact is enough, "
    "two when two facts must be combined. "
    "Pick a tool for each sub-question:\n"
    "page = look up a specific named page (an event, a Games edition, a country at a Games);\n"
    "graph = follow links between entities (people, places, Games, countries);\n"
    "vector = general semantic search.\n"
    'Return ONLY JSON like [{"q": "...", "tool": "page"}]'
)

STOPWORDS = {"the", "a", "an", "of", "at", "in", "on", "and", "or", "for", "to", "by", "with", "from", "is", "was",
             "were", "are", "who", "what", "which", "when", "where", "how", "did", "do", "does", "that", "this",
             "won", "win", "many", "name", "list", "between", "than", "also", "both", "has", "had", "have"}


# =====================================================================
# TOOL 1: vector search
# =====================================================================
def tool_vector(q, k=4):
    return rp.retrieve(q, k=k)


# =====================================================================
# TOOL 2: graph retrieval (TigerGraph hybrid query; falls back to the local entity graph)
# =====================================================================
def tool_graph(q):
    return rp.tg_hits(q)


# =====================================================================
# TOOL 3: page lookup by title (lexical, IDF-weighted; costs zero LLM tokens)
# =====================================================================
_TITLES = None


def _words(s):
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOPWORDS]


def title_index():
    global _TITLES
    if _TITLES is None:
        metas = rp.collection().get(include=["metadatas"])["metadatas"]
        titles = sorted({m["title"] for m in metas if m.get("title")})
        words = {t: set(_words(t)) for t in titles}
        df = Counter(w for ws in words.values() for w in ws)
        _TITLES = (titles, words, df)
    return _TITLES


def match_titles(q, top=2, min_score=0.5):
    """Rank page titles by IDF-weighted word overlap with the question: [(score, title), ...]."""
    titles, words, df = title_index()
    n = len(titles)

    def idf(w):
        return math.log(1 + n / df.get(w, 1))

    qw = set(_words(q))
    q_mass = sum(idf(w) for w in qw if w in df) or 1.0
    scored = []
    for t in titles:
        tw = words[t]
        inter = qw & tw
        if not inter:
            continue
        shared = sum(idf(w) for w in inter)
        scored.append((0.7 * shared / sum(idf(w) for w in tw) + 0.3 * shared / q_mass, t))
    scored.sort(reverse=True)
    return [(s, t) for s, t in scored[:top] if s >= min_score]


def tool_page(q, k_pages=2, k_chunks=3):
    qv = rp.embed_query(q)
    col = rp.collection()
    hits = []
    for _, title in match_titles(q, top=k_pages):
        got = col.get(where={"title": title}, include=["documents", "metadatas", "embeddings"])
        if not got["ids"]:
            continue
        sims = np.array(got["embeddings"]) @ qv
        order = list(np.argsort(-sims)[:k_chunks])
        first = min(range(len(got["ids"])), key=lambda i: got["metadatas"][i]["chunk_idx"])
        if first not in order:                       # always keep the lead/infobox chunk of the page
            order = [first] + order[:k_chunks - 1]
        for i in order:
            m = got["metadatas"][i]
            hits.append(rp.Hit(got["ids"][i], m["doc_id"], m["title"], m["chunk_idx"],
                               got["documents"][i], float(sims[i])))
    return hits


TOOLS = {"page": tool_page, "graph": tool_graph, "vector": tool_vector}


# =====================================================================
# PLANNER
# =====================================================================
def default_tool(q):
    return "page" if match_titles(q, top=1, min_score=0.6) else "graph"


def parse_plan(text, question):
    """Turn the planner's JSON into [{'q','tool'}...]; any malformed output degrades to a one-step plan."""
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    plan = []
    for it in items[:MAX_HOPS]:
        if isinstance(it, dict) and isinstance(it.get("q"), str) and it["q"].strip():
            tool = it.get("tool") if it.get("tool") in TOOLS else default_tool(it["q"])
            plan.append({"q": it["q"].strip(), "tool": tool})
        elif isinstance(it, str) and it.strip():
            plan.append({"q": it.strip(), "tool": default_tool(it)})
    return plan or [{"q": question, "tool": default_tool(question)}]


# =====================================================================
# EXECUTION
# =====================================================================
def _merge(*hit_lists):
    best = {}
    for hs in hit_lists:
        for h in hs:
            if h.id not in best or h.score > best[h.id].score:
                best[h.id] = h
    return sorted(best.values(), key=lambda h: -h.score)


def _safe(fn, q):
    try:
        return fn(q)
    except Exception as e:                            # one failing tool must not kill the investigation
        rp.log(f"  tool error ({fn.__name__}): {str(e)[:70]}")
        return []


def run_hop(sq, tool):
    hits = _safe(TOOLS[tool], sq)
    rp.log(f"  [{tool}] -> {len(hits)} chunks")
    ans = "INSUFFICIENT"
    if hits:
        best = rp.rerank(sq, hits, HOP_TOP_N)
        ans = rp.answer_from_context(sq, rp.build_context(sq, best, HOP_BUDGET), max_tokens=50)
    if rp.is_insufficient(ans):                       # reflect: widen once, using every tool
        wide = _merge(tool_vector(sq, k=8), _safe(tool_graph, sq), _safe(tool_page, sq))
        rp.log(f"  reflect: '{tool}' was not enough -> widened search ({len(wide)} chunks)")
        if wide:
            best = rp.rerank(sq, wide, WIDE_TOP_N)
            ans = rp.answer_from_context(sq, rp.build_context(sq, best, WIDE_BUDGET), max_tokens=50)
    return ans


def agentic_graphrag(q):
    spec = facts.parse(q)
    if spec:                                          # counting / ranking / lookup by venue+date: compute, don't guess
        rp.log(f"plan: structured {spec['kind']} question -> [table] tool over parsed infobox facts (no planner call)")
        a = facts.answer(q)
        if a and not rp.is_insufficient(a):
            rp.log(f"  answer: {a}")
            return a
        rp.log("  table tool had no usable answer -> continuing with the retrieval agent")
    label = rp.current.get("route") or rp.route(q)
    if label == "single":                             # one fact needed: planning would only burn tokens
        plan = [{"q": q, "tool": default_tool(q)}]
        rp.log(f"plan: single-hop question -> no planner call, one [{plan[0]['tool']}] step")
    else:
        plan = parse_plan(rp.llm([{"role": "system", "content": PLAN_SYSTEM},
                                  {"role": "user", "content": q}], max_tokens=150), q)
        rp.log("plan: " + " | ".join(f"[{s['tool']}] {s['q']}" for s in plan))
    answers, fact_lines = [], []
    for i, step in enumerate(plan, 1):
        sq = step["q"]
        for j, a in enumerate(answers, 1):
            sq = sq.replace(f"#{j}", a)
        rp.log(f"hop {i}: {sq}")
        a = run_hop(sq, step["tool"])
        rp.log(f"  answer: {a}")
        if rp.is_insufficient(a):
            return "INSUFFICIENT"
        answers.append(a)
        fact_lines.append(f"- {sq} => {a}")
    if len(plan) == 1:
        return answers[0]
    rp.log("synthesize final answer from hop answers")
    return rp.answer_from_context(q, "\n".join(fact_lines), max_tokens=80)
