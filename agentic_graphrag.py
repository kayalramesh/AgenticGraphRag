"""
Olympics QA: RAG vs GraphRAG vs Agentic GraphRAG  (one file, three pipelines, same LLM, same corpus)
=====================================================================================================
    rag        Pipeline 1  similarity search -> pack -> LLM                                  (rag_pipeline.single_hop_rag)
    graphrag   Pipeline 2  EVENT-GRAPH retrieval (Games -> Sport -> Event nodes with attributes) + LLM reads the
                           compact evidence; non-templated questions fall back to TigerGraph/local entity-graph hybrid
    agentic    Pipeline 3  orchestrator agent: playbooks + LLM planner over specialised tools, EXACT computation
                           (count / argmax / lookup / temporal hop) in tools, evidence judge, cited answer
 
Commands
    python agentic_graphrag.py ask "question" [--pipeline rag|graphrag|agentic|all]
    python agentic_graphrag.py compare [--file eval_public.jsonl] [--limit N]        # accuracy/tokens/latency per pipeline + per question type
    python agentic_graphrag.py run --file eval_hidden.jsonl --pipeline agentic --out answers_hidden.jsonl
    python agentic_graphrag.py show "Men's 200 metres"                               # print the chunks of a page (inspect data format)
 
Agent harness: State (question, linked entities, evidence, facts, step log) | Orchestrator (LLM JSON planner, next action
depends on state) | Tools: link, search, graph, doc, bridge, hop, table, lookup, event, temporal, (tg) | Evidence judge |
Stopping: exact tool answer, judge sufficient, max steps, token budget, 2 steps without new evidence.
"""
import argparse
import difflib
import inspect
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
 
import numpy as np
from tqdm import tqdm
 
import rag_pipeline as rp
 
# ---- token / context-window profile applied to ALL THREE pipelines (rag_pipeline reads these at call time) ----
# One num_ctx for every call => Ollama never reloads the model; 3072 fits a 7B fully in GPU memory (4096 spilled to CPU).
rp.CTX_WINDOW = 3072
rp.P1_CONTEXT_BUDGET = 1100      # RAG evidence tokens (was 900; prompt+answer still < 1.5k)
rp.P3_CONTEXT_BUDGET = 1100      # entity-graph fallback evidence tokens (was 1300)
rp.HOP_CONTEXT_BUDGET = 450      # per-hop evidence tokens (was 500)
 
AGENT_CTX = rp.CTX_WINDOW
MAX_STEPS = 5
TOKEN_BUDGET = 6000              # prompt+completion tokens per question (agent stops and answers when exceeded)
FINAL_CTX_TOKENS = 1800          # evidence tokens in the agent's answering call
FINAL_MAX_TOKENS = 200
PIPES = ["rag", "graphrag", "agentic"]
 
 
# =====================================================================
# LLM helper (Ollama client + token meter shared with rag_pipeline)
# =====================================================================
def _chat(messages, max_tokens=200, json_mode=False):
    opts = {"temperature": 0, "num_predict": max_tokens, "num_ctx": AGENT_CTX}
    if rp.LLM_NUM_GPU is not None:
        opts["num_gpu"] = rp.LLM_NUM_GPU
    kw = {"format": "json"} if json_mode else {}
    for attempt in range(3):
        try:
            r = rp.client.chat(model=rp.LLM_MODEL, messages=messages, options=opts, **kw)
            break
        except rp.ollama.ResponseError:
            if attempt == 2:
                raise
            time.sleep(2)
    rp.meter["prompt"] += r.get("prompt_eval_count", 0) or 0
    rp.meter["completion"] += r.get("eval_count", 0) or 0
    rp.meter["calls"] += 1
    return r["message"]["content"].strip()
 
 
def _json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except Exception:
            return {}
 
 
def _used_tokens():
    return rp.meter["prompt"] + rp.meter["completion"]
 
 
# =====================================================================
# Evidence + corpus indexes
# =====================================================================
@dataclass
class Ev:
    id: str
    title: str
    text: str
    score: float
    src: str
    ans: str = ""                                   # exact answer computed by a tool (agentic only)
    plain: str = ""                                 # evidence WITHOUT the computed answer (GraphRAG pipeline)
    cites: list = field(default_factory=list)       # [(title, chunk_id)]
 
 
_corpus = None
_meta = None
_ent_lc = None
_events = None
 
 
def corpus():
    """chunk id -> (title, doc_id, text, chunk_idx) for the whole collection."""
    global _corpus
    if _corpus is None:
        col, n, c = rp.collection(), rp.collection().count(), {}
        for off in range(0, n, 5000):
            g = col.get(limit=5000, offset=off, include=["documents", "metadatas"])
            for i, d, m in zip(g["ids"], g["documents"], g["metadatas"]):
                c[i] = (m["title"], m["doc_id"], d, int(m.get("chunk_idx", 0)))
        _corpus = c
    return _corpus
 
 
def meta():
    global _meta
    if _meta is None:
        C = corpus()
        _meta = {"doc": {i: v[1] for i, v in C.items()}, "lower": {i: v[0].lower() for i, v in C.items()}}
    return _meta
 
 
def ent_lc():
    global _ent_lc
    if _ent_lc is None:
        _ent_lc = {e.lower(): e for e in rp.entity_index()}
    return _ent_lc
 
 
def fetch(ids, src, score=0.5):
    out = []
    ids = list(ids)
    for i in range(0, len(ids), 500):
        got = rp.collection().get(ids=ids[i:i + 500], include=["documents", "metadatas"])
        for cid, d, m in zip(got["ids"], got["documents"], got["metadatas"]):
            out.append(Ev(cid, m["title"], d, score, src))
    return out
 
 
def rank_ids(ids, question, k, src):
    ids = list(ids)[:400]
    if not ids:
        return []
    got = rp.collection().get(ids=ids, include=["embeddings", "documents", "metadatas"])
    sims = np.array(got["embeddings"]) @ rp.embed_query(question)
    return [Ev(got["ids"][i], got["metadatas"][i]["title"], got["documents"][i], float(sims[i]), src)
            for i in np.argsort(-sims)[:k]]
 
 
def title_terms(s):
    return [t.strip().lower() for t in re.split(r"[|;]", s) if t.strip()]
 
 
STOP = {"the", "of", "and", "at", "in", "on", "a", "to", "olympics", "olympic", "event", "events"}
 
 
def words(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)           # "TechnologyUniversity" -> "Technology University"
    return [w for w in re.split(r"\W+", s.lower()) if w and w not in STOP]
 
 
def norm_title(s):
    return re.sub(r"\s+", " ", re.sub(r"[\u2013\u2014\u2212-]", "-", s.lower())).strip()
 
 
# ---------------- event graph: Games -> Sport -> Event nodes parsed from page titles ----------------
EV_RX = re.compile(r"^(?P<sport>.+?) at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics\s*[\u2013\u2014-]\s*(?P<event>.+)$")
BOUND = r"(?-i:(?=\s+[a-z][a-z_]*:\s|\s*$))"          # a value ends where the next 'key:' starts
FIELD_KEYS = {"venue": ("venue",), "dates": ("dates", "date"), "competitors": ("competitors",), "nations": ("nations",),
              "gold": ("gold", "gold_medalist", "gold_medalists", "gold_medal", "winner", "winners")}
 
 
def get_field(txt, *names):
    for n in names:
        m = re.search(rf"\b{n}:\s*(.+?){BOUND}", txt, re.I | re.S)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""
 
 
def events():
    """One record per event page (doc): sport, year, season, event name, chunk ids. Attributes parsed lazily."""
    global _events
    if _events is None:
        C, by_doc, out = corpus(), defaultdict(list), []
        for cid, (t, d, x, i) in C.items():
            by_doc[d].append((i, cid))
        for d, lst in by_doc.items():
            lst.sort()
            t = C[lst[0][1]][0]
            m = EV_RX.match(t)
            if m:
                out.append({"doc": d, "title": t, "norm": norm_title(t), "sport": m["sport"].lower(), "year": m["year"],
                            "season": m["season"].lower(), "event": m["event"], "cids": [c for _, c in lst]})
        _events = out
    return _events
 
 
def fields(e):
    if "f" not in e:
        C, f = corpus(), {}
        for cid in e["cids"][:4]:
            txt = C[cid][2]
            for k, names in FIELD_KEYS.items():
                if k not in f:
                    v = get_field(txt, *names)
                    if v:
                        f[k] = v
        e["f"] = f
    return e["f"]
 
 
def num(v):
    m = re.match(r"\d[\d,]*", v or "")
    return int(m.group(0).replace(",", "")) if m else None
 
 
def head_text(e, n_words=130):
    return " ".join(corpus()[e["cids"][0]][2].split()[:n_words])
 
 
def page_ev(e, st=None, ans="", gold=""):
    """Evidence for one event page. `plain` = header only; `text` also carries the parsed GOLD line."""
    C = corpus()
    base = head_text(e)
    extra = ""
    if not gold and st is not None and len(e["cids"]) > 1:      # winner not parsed: give the LLM the most relevant chunks
        for x in rank_ids(e["cids"][1:], st.q, 2, "event"):
            extra += " " + " ".join(x.text.split()[:100])
    plain = base + extra
    text = (f"GOLD: {gold}. " if gold else "") + plain
    return Ev(e["cids"][0], e["title"], text, 1.0, "event", ans=ans, plain=plain, cites=[(e["title"], e["cids"][0])])
 
 
# =====================================================================
# SPECIALISED AGENTS / TOOLS   (each returns (observation_text, [Ev]))
# =====================================================================
def link(mention):
    idx, lc, m = rp.entity_index(), ent_lc(), mention.strip().lower()
    if m in lc:
        names = [lc[m]]
    else:
        names = [orig for low, orig in lc.items() if m in low][:300]
        names.sort(key=lambda k: -difflib.SequenceMatcher(None, m, k.lower()).ratio())
        if not names:
            pool = [orig for orig in lc.values() if abs(len(orig) - len(mention)) <= 4]
            names = difflib.get_close_matches(mention, pool, n=3, cutoff=0.8)
    return [(n, len(idx[n])) for n in names[:3]]
 
 
def t_link(st, mention):
    res = link(mention)
    if not res:
        return f"no entity matches '{mention}'", []
    st.entities[mention] = res
    return "linked: " + ", ".join(f"{n}({c} chunks)" for n, c in res), []
 
 
def _hits_to_ev(res, src):
    return [Ev(cid, m["title"], doc, 1 - dist, src)
            for cid, doc, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]
 
 
def t_search(st, query, k=6, contains=""):
    """Similarity search; `contains` keeps only pages whose title contains ALL |-separated terms."""
    k = max(1, min(int(k), 10))
    if not contains:
        items = [Ev(h.id, h.title, h.text, h.score, "search") for h in rp.retrieve(query, k)]
    else:
        terms, M = title_terms(contains), meta()
        dids = list({M["doc"][i] for i, t in M["lower"].items() if all(x in t for x in terms)})
        if not dids:
            return f"no pages with title containing {terms}", []
        qv = rp.embed_query(query).tolist()
        try:
            res = rp.collection().query(query_embeddings=[qv], n_results=k, where={"doc_id": {"$in": dids}})
            items = _hits_to_ev(res, "search")
        except Exception:
            res = rp.collection().query(query_embeddings=[qv], n_results=400)
            items = [e for e in _hits_to_ev(res, "search") if all(x in e.title.lower() for x in terms)][:k]
    return f"{len(items)} chunks: " + "; ".join(sorted({e.title[:60] for e in items})[:4]), items
 
 
def t_graph(st, entity, contains="", k=5):
    found = link(entity)
    if not found:
        return f"entity '{entity}' not in graph", []
    name = found[0][0]
    ids = set(rp.entity_index()[name])
    if contains:
        terms, M = title_terms(contains), meta()
        ids = {i for i in ids if all(x in M["lower"][i] for x in terms)}
    if not ids:
        return f"'{name}' has no chunks matching {contains!r}", []
    items = rank_ids(ids, st.q, max(1, min(int(k), 8)), "graph")
    return f"{name}: {len(ids)} linked chunks, kept {len(items)}: " + "; ".join(sorted({e.title[:60] for e in items})[:4]), items
 
 
def t_doc(st, title, k=4):
    terms, M = title_terms(title), meta()
    ids = [i for i, t in M["lower"].items() if all(x in t for x in terms)]
    if not ids:
        return f"no page title contains {terms}", []
    items = rank_ids(ids, st.q, max(1, min(int(k), 8)), "doc")
    return f"{len(ids)} chunks in matching pages, read {len(items)}: " + "; ".join(sorted({e.title[:60] for e in items})[:3]), items
 
 
BAD_NAME = re.compile(r"(Olympic|Games|Summer|Winter|Championship|Stadium|Cup|Men|Women|Team|Federation|Committee|"
                      r"National|University|Association|Union|Republic|Kingdom|States|Beijing|London|Tokyo|Rio|Athens|Sydney)", re.I)
MEDAL = re.compile(r"\b(gold|silver|bronze|medal\w*|champion\w*|won|winner|podium)\b", re.I)
 
 
def personish(e):
    p = e.split()
    return 2 <= len(p) <= 4 and not BAD_NAME.search(e) and all(re.fullmatch(r"[A-Z][a-zA-Z'’\-\.]+", t) for t in p)
 
 
def windows(text, name, w=160):
    out = []
    for m in re.finditer(re.escape(name), text):
        s = text[max(0, m.start() - w): m.end() + w]
        if MEDAL.search(s):
            out.append(" ".join(s.split()))
    return out
 
 
def t_bridge(st, a, b, top=8):
    """Aggregation over the entity graph: entities in pages matching `a` AND `b`, verified by result wording in both."""
    M, idx = meta(), rp.entity_index()
    ta, tb = title_terms(a), title_terms(b)
    A = {i for i, t in M["lower"].items() if all(x in t for x in ta)}
    B = {i for i, t in M["lower"].items() if all(x in t for x in tb)}
    if not A or not B:
        return f"bridge: no pages for {'A' if not A else 'B'} ({a!r} / {b!r})", []
    cands = []
    for e, ids in idx.items():
        if len(ids) > rp.HUB_CAP or not personish(e):
            continue
        s = set(ids)
        ia, ib = s & A, s & B
        if ia and ib:
            cands.append((min(len(ia), len(ib)), e, ia, ib))
    cands.sort(key=lambda x: -x[0])
    cands = cands[:60]
    if not cands:
        return "bridge: no shared person-like entities", []
    texts = {e.id: e.text for e in fetch({i for _, _, ia, ib in cands for i in (ia | ib)}, "bridge")}
    verified = []
    for _, name, ia, ib in cands:
        wa = [w for i in ia for w in windows(texts.get(i, ""), name)]
        wb = [w for i in ib for w in windows(texts.get(i, ""), name)]
        if wa and wb:
            verified.append((len(wa) + len(wb), name, wa[0], wb[0]))
    verified.sort(key=lambda x: -x[0])
    items = [Ev(f"bridge::{n}", f"{n} ({a} & {b})", f"{n}. In '{a}': ...{wa[:260]}... In '{b}': ...{wb[:260]}...", 1.0, "bridge")
             for _, n, wa, wb in verified[:top]]
    return (f"{len(cands)} shared candidates, {len(verified)} verified; top: {', '.join(n for _, n, _, _ in verified[:top])}"), items
 
 
def t_hop(st, subquestion, k=4):
    hits = rp.retrieve(subquestion, k=k)
    items = [Ev(h.id, h.title, h.text, h.score, "hop") for h in hits]
    ctx = rp.build_context(subquestion, hits, 500)
    a = _chat([{"role": "system", "content": rp.SYSTEM},
               {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {subquestion}\nAnswer:"}], 50)
    if rp.is_insufficient(a):
        return f"sub-question not answerable from top chunks: {subquestion}", items
    st.facts.append(f"{subquestion} => {a}")
    return f"fact: {subquestion} => {a}", items
 
 
def t_tg(st, query):
    from tg_graphrag import tg_retrieve
    hits = tg_retrieve(query)
    items = [Ev(h.id, h.title, h.text, h.score, "tg") for h in hits]
    return f"{len(items)} chunks from TigerGraph: " + "; ".join(sorted({e.title[:60] for e in items})[:4]), items
 
 
# ---------------- structured tools over the event graph (exact; no LLM) ----------------
def games_parts(s):
    y = re.search(r"\b(\d{4})\b", s or "")
    se = re.search(r"summer|winter", s or "", re.I)
    return (y.group(1) if y else "", se.group(0).lower() if se else "")
 
 
def t_table(st, sport, games, field="competitors", gt="", lt=""):
    """Aggregation / superlative: read `field` (competitors|nations) of EVERY event of a sport at a Games and
    count / rank in Python (an LLM cannot count 40 numbers reliably)."""
    year, season = games_parts(games)
    sp = re.sub(r"\bevents?\b", "", sport.lower()).strip()
    rows = []
    for e in events():
        if (not year or e["year"] == year) and (not season or e["season"] == season) and (e["sport"] == sp or sp in e["sport"]):
            v = num(fields(e).get(field))
            if v is not None:
                rows.append((e, v))
    if not rows:
        return f"no event pages with '{field}' for {sport!r} at {games!r}", []
    rows.sort(key=lambda x: (-x[1], x[0]["title"]))
    best_e, best_v = rows[0]
    summary = f"{len(rows)} events read. MAX {field} = {best_v}: {best_e['title']}."
    sel, ans = rows[:15], best_e["title"]
    if str(gt).strip():
        sel = [(e, v) for e, v in rows if v > int(gt)]
        ans = str(len(sel))
        summary += f" COUNT of events with {field} > {gt}: {ans}."
    elif str(lt).strip():
        sel = [(e, v) for e, v in rows if v < int(lt)]
        ans = str(len(sel))
        summary += f" COUNT of events with {field} < {lt}: {ans}."
    text = summary + "\n" + "\n".join(f"{e['title']} | {field}={v}" for e, v in sel[:25])
    stem = best_e["title"].split(" \u2013 ")[0]
    prefix = f"Page title format: '{stem} \u2013 <event>'"
    plain = f"{prefix}. {field} per event:\n" + "\n".join(f"{e['event']} = {v}" for e, v in sorted(rows, key=lambda x: x[0]["title"])[:60])
    return summary, [Ev(f"table::{sp}::{year}{season}::{field}", f"Event table: {sp} at {year} {season}", text, 1.0, "table",
                        ans=ans, plain=plain, cites=[(e["title"], e["cids"][0]) for e, _ in sel[:12]])]
 
 
def t_lookup(st, title, field="nations"):
    """Attribute of one named event page (nations / competitors / venue / dates)."""
    nt = norm_title(title)
    cand = [e for e in events() if e["norm"] == nt]
    if not cand:
        close = difflib.get_close_matches(nt, [e["norm"] for e in events()], n=1, cutoff=0.88)
        cand = [e for e in events() if close and e["norm"] == close[0]]
    if not cand:
        return f"no event page titled {title!r}", []
    e = cand[0]
    v = fields(e).get(field, "")
    ans = str(num(v)) if field in ("nations", "competitors") and num(v) is not None else v
    ev = page_ev(e, st, ans=ans, gold="x" if ans else "")
    ev.text = (f"{field.upper()} = {ans}. " if ans else "") + ev.plain
    return f"{e['title']}: {field}={ans or 'n/a'}", [ev]
 
 
def _best_event(sport_games_phrase, year, season):
    toks = set(words(sport_games_phrase))
    best, bs = None, 0.0
    for e in events():
        if e["year"] != str(year) or e["season"] != season:
            continue
        tt = set(words(e["title"]))
        sc = len(toks & tt) / max(len(toks), 1)
        if sc > bs or (sc == bs and best is not None and len(tt) < len(set(words(best["title"])))):
            best, bs = e, sc
    return (best, bs) if bs >= 0.75 else (None, bs)
 
 
def _gold_ev(e, st):
    g = fields(e).get("gold", "")
    g = g if g and len(g.split()) <= 14 else ""
    return page_ev(e, st, ans=g, gold=g)
 
 
def t_event(st, venue, date="", games=""):
    """Find the event page by venue + date (multi-hop: venue/date -> page -> gold medallist)."""
    vt, dt = set(words(venue)), set(words(date))
    year, season = games_parts(games or date)
    scored = []
    for e in events():
        if (year and e["year"] != year) or (season and e["season"] != season):
            continue
        f = fields(e)
        if "venue" not in f:
            continue
        vs = len(vt & set(words(f["venue"]))) / max(len(vt), 1)
        if vs < 0.6:
            continue
        ds = len(dt & set(words(f.get("dates", "")))) / max(len(dt), 1) if dt else 0
        scored.append((vs * 2 + ds * 3, e))
    if not scored:
        return f"no event page with venue like {venue!r}", []
    scored.sort(key=lambda x: -x[0])
    e = scored[0][1]
    ev = _gold_ev(e, st)
    amb = f" (+{len(scored) - 1} other candidates)" if len(scored) > 1 and scored[1][0] == scored[0][0] else ""
    return f"{e['title']} at {fields(e).get('venue', '')} [{fields(e).get('dates', '')}] gold={ev.ans or 'not parsed'}{amb}", [ev]
 
 
SUMMER = [1896, 1900, 1904, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980,
          1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]
WINTER = [1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1994, 1998, 2002,
          2006, 2010, 2014, 2018, 2022]
 
 
def t_temporal(st, event, season, before):
    """Temporal hop: 'the Games held immediately before YEAR' -> that year's page for the event -> gold medallist."""
    season = season.lower()
    prev = max([y for y in (SUMMER if season == "summer" else WINTER) if y < int(before)], default=None)
    if prev is None:
        return f"no {season} Games before {before}", []
    e, sc = _best_event(event, prev, season)
    if e is None:
        return f"{season} {prev}: no event page matches {event!r} (best score {sc:.2f})", []
    ev = _gold_ev(e, st)
    return f"immediately before {before} = {prev}; page {e['title']}; gold={ev.ans or 'not parsed'}", [ev]
 
 
TOOLS = {"link": t_link, "search": t_search, "graph": t_graph, "doc": t_doc, "bridge": t_bridge, "hop": t_hop,
         "table": t_table, "lookup": t_lookup, "event": t_event, "temporal": t_temporal, "tg": t_tg}
 
# ---------------- playbooks: regex recognisers for question templates (0 LLM tokens to plan) ----------------
PLAYBOOKS = [
    (re.compile(r"how many (?P<sport>.+?) events at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics had "
                r"(?P<cmp>more|fewer|less) than (?P<n>\d+) (?P<field>competitors|nations)", re.I),
     lambda m: ("table", {"sport": m["sport"], "games": f"{m['year']} {m['season']}", "field": m["field"].lower(),
                          ("gt" if m["cmp"].lower() == "more" else "lt"): m["n"]})),
    (re.compile(r"which (?P<sport>.+?) event at the (?P<year>\d{4}) (?P<season>Summer|Winter) Olympics had the "
                r"(?:highest|most|largest|lowest|fewest|smallest) number of (?P<field>competitors|nations)", re.I),
     lambda m: ("table", {"sport": m["sport"], "games": f"{m['year']} {m['season']}", "field": m["field"].lower()})),
    (re.compile(r"how many (?P<field>nations|competitors) (?:competed|took part|participated) in (?P<title>.+?)\s*\?*\s*$", re.I),
     lambda m: ("lookup", {"title": m["title"], "field": m["field"].lower()})),
    (re.compile(r"who won the gold medal in the (?P<ev>.+?) event at the (?P<season>Summer|Winter) Olympics held immediately "
                r"before (?P<y>\d{4})", re.I),
     lambda m: ("temporal", {"event": m["ev"], "season": m["season"], "before": m["y"]})),
    (re.compile(r"held at (?P<venue>.+?) (?:on|between) (?P<date>.+?)(?: at the (?P<games>\d{4} (?:Summer|Winter)) Olympics)?\s*\??\s*$", re.I),
     lambda m: ("event", {"venue": m["venue"], "date": m["date"], "games": m["games"] or ""})),
]
 
 
def playbook(q):
    for rx, build in PLAYBOOKS:
        m = rx.search(q)
        if m:
            return [build(m)]
    return []
 
 
TOOL_DOC = """link(mention)                     entity linking: map a name to corpus entities (+ chunk counts)
search(query, k=6, contains="")  similarity search; contains="2012 Summer" keeps pages whose title has it ("a|b" = both)
graph(entity, contains="")       graph traversal: chunks that mention the entity, ranked by the question
doc(title)                       document retrieval: read chunks of pages whose title contains the text
bridge(a, b)                     aggregation: people in pages containing a AND b with result wording (for "both/also" questions)
table(sport, games, field="competitors", gt="", lt="")  exact count/max over EVERY event of a sport at a Games, e.g. sport="biathlon", games="2018 Winter"
lookup(title, field="nations")   attribute of one event page, e.g. title="Judo at the 2016 Summer Olympics \u2013 Women's 57 kg"
event(venue, date="", games="")  event page held at a venue/date -> its gold medallist
temporal(event, season, before)  gold medallist of an event at the Games immediately before a year, e.g. event="men's pole vault athletics", season="Summer", before=2016
hop(subquestion)                 multi-hop: answer one sub-question and store the fact
finish()                         the evidence is enough: answer now"""
TG_DOC = "\ntg(query)                        TigerGraph hybrid GraphRAG retrieval (slow); use if search+graph keep missing"
 
ORCH_SYS = ("You are the orchestrator of an investigation over an Olympics Wikipedia corpus. After every observation "
            "choose ONE next action. Reply with JSON only: {\"thought\": \"<short>\", \"action\": \"<tool>\", \"args\": {...}}.\n"
            "Tools:\n{tools}\n"
            "Rules: counts/highest per sport -> table. one named event -> lookup. 'event held at <venue>' -> event. "
            "'Games immediately before' -> temporal. 'both/also' -> bridge. Chains -> hop or search then graph. "
            "Never repeat an identical call. Use finish as soon as the evidence answers the question.")
 
 
# =====================================================================
# HARNESS: state, orchestrator, judge, final answer
# =====================================================================
@dataclass
class State:
    q: str
    entities: dict = field(default_factory=dict)
    ev: dict = field(default_factory=dict)
    facts: list = field(default_factory=list)
    log: list = field(default_factory=list)
    done: set = field(default_factory=set)
    overrides: int = 0
    stagnant: int = 0
 
    def add(self, items):
        new = 0
        for e in items:
            old = self.ev.get(e.id)
            if old is None:
                self.ev[e.id] = e
                new += 1
            elif e.score > old.score:
                self.ev[e.id] = e
        return new
 
 
def state_view(st, hint):
    lines = [f"Question: {st.q}"]
    if st.entities:
        lines.append("Linked entities: " + "; ".join(f"{m}->{r[0][0]}({r[0][1]})" for m, r in st.entities.items() if r))
    if st.facts:
        lines.append("Facts found:\n" + "\n".join(f"- {f}" for f in st.facts))
    if st.log:
        lines.append("Steps so far:\n" + "\n".join(
            f"{s['step']}. {s['action']}({json.dumps(s['args'], ensure_ascii=False)}) -> {s['observation'][:220]}" for s in st.log))
    top = sorted(st.ev.values(), key=lambda e: -e.score)[:5]
    lines.append(f"Evidence held: {len(st.ev)} chunks." + ("\n" + "\n".join(f"- [{e.title[:70]}] {e.text[:120]}" for e in top) if top else ""))
    if hint:
        lines.append(f"Note: {hint}")
    lines.append("Next action (JSON):")
    return "\n".join(lines)
 
 
def default_action(st):
    if not st.ev:
        return "search", {"query": st.q}
    if not any(s["action"] == "graph" for s in st.log):
        for r in st.entities.values():
            if r:
                return "graph", {"entity": r[0][0]}
    return "finish", {}
 
 
def plan_step(st, hint, tools_doc):
    out = _chat([{"role": "system", "content": ORCH_SYS.replace("{tools}", tools_doc)},
                 {"role": "user", "content": state_view(st, hint)}], 160, json_mode=True)
    d = _json(out)
    act, args = d.get("action"), d.get("args") if isinstance(d.get("args"), dict) else {}
    if act != "finish" and act not in TOOLS:
        act, args = default_action(st)
        return act, args, "(fallback policy)"
    return act, args, str(d.get("thought", ""))[:160]
 
 
def run_tool(st, act, args):
    fn = TOOLS[act]
    params = inspect.signature(fn).parameters
    clean = {k: v for k, v in args.items() if k in params and k != "st"}
    required = [p for p in list(params)[1:] if params[p].default is inspect.Parameter.empty]
    if any(r not in clean for r in required):
        if "query" in required:
            clean["query"] = st.q
        else:
            return f"missing argument(s) for {act}: {required}", []
    try:
        return fn(st, **clean)
    except Exception as e:
        return f"{act} failed: {str(e)[:120]}", []
 
 
def judge(st):
    top = sorted(st.ev.values(), key=lambda e: (-(e.src in ("bridge", "table", "event")), -e.score))[:6]
    view = "\n".join(f"- [{e.title[:70]}] {e.text[:350]}" for e in top)
    facts = "\n".join(f"- {f}" for f in st.facts)
    d = _json(_chat([
        {"role": "system", "content": "You judge evidence for a question. Reply JSON only: "
                                      "{\"sufficient\": true|false, \"missing\": \"what is still needed\"}. "
                                      "sufficient=true only if the evidence/facts directly contain what is needed to answer completely."},
        {"role": "user", "content": f"Question: {st.q}\nFacts:\n{facts}\nEvidence:\n{view}"}], 80, True))
    return {"sufficient": bool(d.get("sufficient", True)), "missing": str(d.get("missing", ""))[:200]}
 
 
def final_answer(st):
    ev = sorted(st.ev.values(), key=lambda e: (-(e.src in ("bridge", "table", "event")), -e.score))
    chosen, lines, used = [], [], 0
    for e in ev:
        if len(chosen) >= 12:
            break
        line = f"[{len(chosen) + 1}] {e.title}: {e.text}"
        c = rp.est_tokens(line)
        if used + c > FINAL_CTX_TOKENS:
            continue
        chosen.append(e)
        lines.append(line)
        used += c
    facts = ("Intermediate facts:\n" + "\n".join(f"- {f}" for f in st.facts) + "\n\n") if st.facts else ""
    raw = _chat([
        {"role": "system", "content": "Answer the question using ONLY the numbered evidence. Be correct and complete: if the "
                                      "question asks for several items, list every item the evidence supports. Give ONLY the answer "
                                      "(a number, a name, a full page title, or a comma-separated list), no explanation, then cite "
                                      "evidence numbers like [1][3]. If the evidence has a line 'COUNT ...' answer with that number; "
                                      "for 'which event' answer with the full page title; for 'who won the gold medal' answer with the "
                                      "winner name(s) only, copied exactly as written. If the evidence does not contain the answer "
                                      "reply exactly: INSUFFICIENT"},
        {"role": "user", "content": f"{facts}Evidence:\n" + "\n".join(lines) + f"\n\nQuestion: {st.q}\nAnswer:"}], FINAL_MAX_TOKENS)
    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", raw) if 0 < int(n) <= len(chosen)})
    cites = [{"n": n, "title": chosen[n - 1].title, "chunk": chosen[n - 1].id} for n in nums]
    return raw, re.sub(r"\s*\[\d+\]", "", raw).strip(), cites
 
 
# =====================================================================
# PIPELINE 3: agentic loop
# =====================================================================
def _result(q, pipeline, ans, cites, st, stop, t0):
    return {"question": q, "route": pipeline, "pipeline": pipeline, "answer": ans, "citations": cites,
            "steps": st.log if st else [], "stop_reason": stop, "evidence_chunks": len(st.ev) if st else 0,
            "prompt_tokens": rp.meter["prompt"], "completion_tokens": rp.meter["completion"],
            "llm_calls": rp.meter["calls"], "seconds": round(time.time() - t0, 2)}
 
 
def agentic(q, max_steps=MAX_STEPS):
    rp.meter.clear()
    rp.trace.clear()
    t0 = time.time()
    st = State(q)
    tools_doc = TOOL_DOC + (TG_DOC if rp.tg_enabled() else "")
    # harness priming (0 LLM tokens): entity linking + playbook recognition
    for act, args in playbook(q):
        obs, items = run_tool(st, act, args)
        st.done.add(act + json.dumps(args, sort_keys=True, ensure_ascii=False))
        st.log.append({"step": 0, "thought": "playbook", "action": act, "args": args, "observation": obs,
                       "new_evidence": st.add(items)})
    exact = next((e for e in st.ev.values() if e.ans), None)
    if exact:                                                       # computed by a tool: no LLM call needed
        rp.trace.append("\n".join(e.text for e in st.ev.values()))
        cites = [{"n": i + 1, "title": t, "chunk": c} for i, (t, c) in enumerate(exact.cites[:12])]
        return _result(q, "agentic", exact.ans, cites, st, "exact_tool", t0)
    for m in sorted(rp.extract_entities(q))[:6]:
        res = link(m)
        if res:
            st.entities[m] = res
    hint, stop = "", "max_steps"
    for step in range(1, max_steps + 1):
        if _used_tokens() > TOKEN_BUDGET:
            stop = "token_budget"
            break
        act, args, thought = plan_step(st, hint, tools_doc)
        if act == "tg" and not rp.tg_enabled():
            act, args = "search", {"query": q}
        if act == "finish":
            if st.ev:
                v = judge(st)
                st.log.append({"step": step, "thought": thought, "action": "finish", "args": {},
                               "observation": f"judge: sufficient={v['sufficient']} missing={v['missing']!r}", "new_evidence": 0})
                if v["sufficient"] or st.overrides >= 1:
                    stop = "judge_sufficient" if v["sufficient"] else "judge_override_limit"
                    break
                st.overrides += 1
                hint = "Evidence judge says still missing: " + v["missing"]
                continue
            act, args = "search", {"query": q}
        key = act + json.dumps(args, sort_keys=True, ensure_ascii=False)
        if key in st.done:
            hint = "You already ran that exact call. Pick a different action or finish."
            continue
        st.done.add(key)
        obs, items = run_tool(st, act, args)
        new = st.add(items)
        st.stagnant = 0 if new else st.stagnant + 1
        st.log.append({"step": step, "thought": thought, "action": act, "args": args, "observation": obs, "new_evidence": new})
        hint = ""
        exact = next((e for e in items if e.ans), None)
        if exact:
            rp.trace.append("\n".join(e.text for e in st.ev.values()))
            cites = [{"n": i + 1, "title": t, "chunk": c} for i, (t, c) in enumerate(exact.cites[:12])]
            return _result(q, "agentic", exact.ans, cites, st, "exact_tool", t0)
        if st.stagnant >= 2:
            stop = "no_new_evidence"
            break
    if not st.ev:
        _, items = t_search(st, q, 8)
        st.add(items)
    raw, ans, cites = final_answer(st)
    if rp.is_insufficient(raw) and len(st.log) < max_steps + 2:
        _, items = t_search(st, q, 10)
        st.add(items)
        raw, ans, cites = final_answer(st)
    rp.trace.append("\n".join(e.text for e in st.ev.values()))
    return _result(q, "agentic", ans if not rp.is_insufficient(raw) else "Not found in corpus", cites, st, stop, t0)
 
 
# =====================================================================
# PIPELINE 2: event-graph retrieval + one LLM call (falls back to TigerGraph/local entity-graph hybrid)
# =====================================================================
def graphrag_structured(q):
    rp.meter.clear()
    rp.trace.clear()
    t0 = time.time()
    st = State(q)
    items = []
    for act, args in playbook(q):
        obs, items = run_tool(st, act, args)
        st.log.append({"step": 0, "thought": "graph expansion", "action": act, "args": args, "observation": obs, "new_evidence": len(items)})
    if not items:                                              # untemplated question: TigerGraph/entity-graph hybrid retrieval
        return rp.ask(q, force="graphrag")
    ctx = "\n".join(f"[{i + 1}] {e.title}: {e.plain or e.text}" for i, e in enumerate(items[:3]))
    rp.trace.append(ctx)
    raw = _chat([
        {"role": "system", "content": "Answer using ONLY the evidence. For counting questions, first list the matching rows, then "
                                      "finish with 'ANSWER: <answer>'. The answer is a number, a name, or a full page title; copy "
                                      "names and titles exactly as written. For every other question reply 'ANSWER: <answer>'. "
                                      "If the evidence lacks it, reply 'ANSWER: INSUFFICIENT'."},
        {"role": "user", "content": f"Evidence:\n{ctx}\n\nQuestion: {q}"}], 170)
    m = re.findall(r"ANSWER:\s*(.+)", raw, re.I)
    ans = (m[-1] if m else raw).strip().rstrip(".")
    cites = [{"n": i + 1, "title": t, "chunk": c} for i, (t, c) in enumerate(items[0].cites[:6])]
    return _result(q, "graphrag", ans, cites, st, "event_graph", t0)
 
 
# =====================================================================
# UNIFIED RUNNER, COMPARISON, SUBMISSION FILES
# =====================================================================
def ask_any(q, pipeline):
    if pipeline == "agentic":
        return agentic(q)
    if pipeline == "graphrag":
        return graphrag_structured(q)
    return rp.ask(q, force="rag")
 
 
def safe_ask(q, pipeline):
    try:
        return ask_any(q, pipeline), None
    except Exception as e:
        rp.trace.clear()
        return ({"answer": "ERROR", "pipeline": pipeline, "prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0,
                 "seconds": 0, "question": q, "citations": [], "steps": []}, str(e)[:150])
 
 
def show(r):
    print(f"\n=== {r['pipeline'].upper()} ===  ({r['seconds']}s, {r['llm_calls']} LLM calls, "
          f"{r['prompt_tokens']}+{r['completion_tokens']} tokens)")
    for s in r.get("steps", []):
        print(f"  step {s['step']}: {s['action']} {json.dumps(s['args'], ensure_ascii=False)}\n          -> {s['observation'][:200]}")
    if r.get("steps"):
        print(f"  stop: {r.get('stop_reason')}, evidence chunks: {r.get('evidence_chunks')}")
    print("ANSWER:", r["answer"])
    for c in r.get("citations", [])[:6]:
        print(f"  [{c['n']}] {c['title']}  ({c['chunk']})")
 
 
def compare(path="eval_public.jsonl", limit=None, pipes=PIPES):
    rows = list(rp.read_jsonl(path))[:limit]
    summary = {}
    for name in pipes:
        ok = tp = tc = calls = errors = maxp = 0
        secs, diag, per_type = 0.0, Counter(), defaultdict(lambda: [0, 0])
        with open(f"eval_results_{name}.jsonl", "w", encoding="utf-8") as f:
            for row in tqdm(rows, desc=name):
                q, gold = row["question"], row["answer"]
                r, err = safe_ask(q, name)
                if err:
                    errors += 1
                    print(f"\n[{name}] ERROR: {err}")
                    if errors >= 3 and ok == 0:
                        print(f"[{name}] 3 errors in a row, skipping this pipeline.")
                        break
                hit = rp.gold_in(rp.trace, gold)
                good = rp.correct(r["answer"], gold)
                ok += good
                tp += r["prompt_tokens"]
                tc += r["completion_tokens"]
                calls += r["llm_calls"]
                secs += r["seconds"]
                maxp = max(maxp, r["prompt_tokens"])
                diag["correct" if good else ("generation_fail" if hit else "retrieval_fail")] += 1
                per_type[row.get("qtype", "?")][0] += good
                per_type[row.get("qtype", "?")][1] += 1
                f.write(json.dumps({**r, "qid": row.get("qid"), "qtype": row.get("qtype"), "gold": gold, "correct": good,
                                    "gold_in_context": hit}, ensure_ascii=False) + "\n")
        n = max(len(rows), 1)
        diag["error"] = errors
        summary[name] = {"accuracy": round(ok / n, 3), "avg_tokens": round((tp + tc) / n), "avg_prompt_tokens": round(tp / n),
                         "max_prompt_tokens": maxp, "avg_llm_calls": round(calls / n, 2), "avg_seconds": round(secs / n, 1),
                         "by_type": {k: f"{v[0]}/{v[1]}" for k, v in per_type.items()}, "failures": dict(diag)}
    print("\n| pipeline | accuracy | avg tokens | max prompt tokens | avg LLM calls | avg sec | failures |\n|---|---|---|---|---|---|---|")
    for k, v in summary.items():
        print(f"| {k} | {v['accuracy']:.1%} | {v['avg_tokens']} | {v['max_prompt_tokens']} | {v['avg_llm_calls']} | {v['avg_seconds']} | {v['failures']} |")
    print("\nAccuracy by question type:")
    for k, v in summary.items():
        print(f"  {k}: {v['by_type']}")
    with open("compare_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved compare_results.json and eval_results_<pipeline>.jsonl")
 
 
def run_file(path, pipeline, out):
    """Answer every question in a file (public 100 / hidden 50). Writes full trace + a minimal submission file."""
    rows = list(rp.read_jsonl(path))
    sub = out.rsplit(".", 1)[0] + "_submission.jsonl"
    errors = 0
    with open(out, "w", encoding="utf-8") as f, open(sub, "w", encoding="utf-8") as g:
        for row in tqdm(rows, desc=f"{pipeline}:{path}"):
            r, err = safe_ask(row["question"], pipeline)
            errors += bool(err)
            f.write(json.dumps({"qid": row.get("qid"), "question": row["question"], "qtype": row.get("qtype"), "pipeline": pipeline,
                                "answer": r["answer"], "citations": r.get("citations", []), "steps": r.get("steps", []),
                                "stop_reason": r.get("stop_reason"), "prompt_tokens": r["prompt_tokens"],
                                "completion_tokens": r["completion_tokens"], "llm_calls": r["llm_calls"],
                                "seconds": r["seconds"], "error": err}, ensure_ascii=False) + "\n")
            g.write(json.dumps({"qid": row.get("qid"), "answer": r["answer"]}, ensure_ascii=False) + "\n")
    print(f"Wrote {out} (full trace) and {sub} (qid + answer). Errors: {errors}/{len(rows)}")
 
 
def show_pages(sub):
    sub = norm_title(sub)
    hit = [e for e in events() if sub in e["norm"]][:2]
    if not hit:
        print("no event page title contains that text")
    for e in hit:
        print(f"\n##### {e['title']}  (doc {e['doc']}, {len(e['cids'])} chunks)  parsed fields: {fields(e)}")
        for cid in e["cids"][:3]:
            print(f"--- {cid}\n{corpus()[cid][2][:1200]}")
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["ask", "compare", "run", "show"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--pipeline", choices=PIPES + ["all"], default="all")
    ap.add_argument("--file", default="eval_public.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a, _ = ap.parse_known_args()
    if a.cmd == "ask":
        for p in (PIPES if a.pipeline == "all" else [a.pipeline]):
            show(ask_any(a.question, p))
    elif a.cmd == "compare":
        compare(a.file, a.limit, PIPES if a.pipeline == "all" else [a.pipeline])
    elif a.cmd == "run":
        for p in (PIPES if a.pipeline == "all" else [a.pipeline]):
            run_file(a.file, p, a.out or f"answers_{a.file.split('/')[-1].split('.')[0]}_{p}.jsonl")
    elif a.cmd == "show":
        show_pages(a.question)
    else:
        print('Commands: ask "q" | compare [--file F --limit N] | run --file F --pipeline P [--out O] | show "title text"')
 