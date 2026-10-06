"""
Structured facts tool ("table") for the agentic pipeline
========================================================
Many questions cannot be answered by reading the top-k chunks, because they need an AGGREGATE over every event page:

    "How many biathlon events at the 2018 Winter Olympics had more than 73 competitors?"   -> count over ~11 pages
    "Which sailing event at the 2000 Summer Olympics had the highest number of competitors?" -> argmax over ~11 pages
    "Who won the gold medal in the event held at Riocentro - Pavilion 4 on 11-19 August ...?" -> find the page by venue+date

A language model guesses at these; code does not. So the agent gets a tool that
  1. parses the flattened infobox at the top of every event page ("event: .. games: .. venue: .. competitors: ..")
     into records (built once from your Chroma index, cached in facts_index.json),
  2. selects the right pages by sport / Games / venue / date and computes the count or ranking,
  3. hands the LLM a small evidence table (a few hundred tokens). The LLM still writes the final answer, and the
     page titles used become the citations.

Debug without any LLM:
    python facts.py build                      # build + show parse quality
    python facts.py ask "<question>"           # which spec was parsed and what evidence the LLM would see
    python facts.py shapes                     # the question templates in eval_public.jsonl
"""
import argparse
import json
import operator
import re
from collections import Counter, defaultdict
from pathlib import Path

import rag_pipeline as rp

FACTS_PATH = "facts_index.json"
GAMES = r"\d{4} (?:Summer|Winter) (?:Olympics|Paralympics)"
EVENT_RE = re.compile(rf"^(?P<sport>.+?) at the (?P<games>{GAMES}) [\u2013\u2014-] (?P<event>.+)$")
KEY_RE = re.compile(r"(?:^|(?<=\s))([a-z][a-z_]{1,24}): ")

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
OP_WORDS = {"more than": ">", "greater than": ">", "over": ">", "at least": ">=", "fewer than": "<",
            "less than": "<", "under": "<", "at most": "<=", "exactly": "=="}
OP_TEXT = {">": "more than", ">=": "at least", "<": "fewer than", "<=": "at most", "==": "exactly"}
CMP = "|".join(sorted(OP_WORDS, key=len, reverse=True))

COUNT_RE = re.compile(
    rf"how many (?P<sport>.+?) events? (?:at|in|during) the (?P<games>{GAMES}) (?:had|have|has|with|featured) "
    rf"(?P<cmp>{CMP}) (?P<n>[\d,]+) (?P<field>competitors|nations)", re.I)
EXTREME_RE = re.compile(
    rf"which (?P<sport>.+?) events? (?:at|in|during) the (?P<games>{GAMES}) (?:had|has|have) the "
    rf"(?P<sup>highest|most|largest|greatest|lowest|fewest|smallest|least)(?: number of)? (?P<field>competitors|nations)", re.I)
LOOKUP_RE = re.compile(
    rf"held at (?P<venue>.+?) (?:on|during) (?P<date>.+?)(?: (?:at|in) the (?P<games>{GAMES}))?[?.\s]*$", re.I)

TABLE_SYSTEM = ("Answer using ONLY the evidence. Give just the answer: a number for 'how many', the full page title "
                "for 'which event', the name(s) exactly as written for 'who'. "
                "If the evidence does not contain the answer, reply exactly: INSUFFICIENT")


# =====================================================================
# 1) PARSE THE FLATTENED INFOBOX OF EVERY EVENT PAGE
# =====================================================================
def parse_fields(text):
    """'event: X games: Y venue: Z competitors: 39 nations: 31 ...' -> {'event': 'X', ...} (first occurrence wins)."""
    text = text[:2500]
    marks = [(m.start(), m.end(), m.group(1)) for m in KEY_RE.finditer(text)]
    fields = {}
    for i, (_, end, key) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        fields.setdefault(key, text[end:stop].strip())
    return fields


def _to_int(v):
    m = re.match(r"\s*([\d,]+)", v or "")
    digits = m.group(1).replace(",", "") if m else ""
    return int(digits) if digits else None


_RECORDS = None


def records(force=False):
    """One record per event page (title like 'Sport at the 2008 Summer Olympics - Event'); cached on disk."""
    global _RECORDS
    if _RECORDS is not None and not force:
        return _RECORDS
    if Path(FACTS_PATH).exists() and not force:
        _RECORDS = json.loads(Path(FACTS_PATH).read_text(encoding="utf-8"))
        return _RECORDS
    col = rp.collection()
    pages = defaultdict(list)
    step = 5000
    for off in range(0, col.count(), step):
        got = col.get(limit=step, offset=off, include=["documents", "metadatas"])
        for doc, m in zip(got["documents"], got["metadatas"]):
            if EVENT_RE.match(m["title"]):
                pages[m["title"]].append((m["chunk_idx"], doc))
    recs = []
    for title, chunks in pages.items():
        chunks.sort()
        f = parse_fields(" ".join(c[1] for c in chunks[:2]))
        mm = EVENT_RE.match(title)
        recs.append({
            "title": title, "sport": mm.group("sport"), "games": mm.group("games"), "event": mm.group("event"),
            "venue": f.get("venue", ""), "dates": f.get("dates") or f.get("date", ""),
            "competitors": _to_int(f.get("competitors")), "nations": _to_int(f.get("nations")),
            "fields": {k: v[:160] for k, v in f.items()},
        })
    Path(FACTS_PATH).write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    _RECORDS = recs
    return recs


# =====================================================================
# 2) UNDERSTAND THE QUESTION (regex, zero tokens)
# =====================================================================
def parse(q):
    """Return a spec dict for aggregation / lookup questions, else None."""
    m = COUNT_RE.search(q)
    if m:
        return {"kind": "count", "sport": m.group("sport"), "games": m.group("games"), "field": m.group("field").lower(),
                "op": OP_WORDS[m.group("cmp").lower()], "n": int(m.group("n").replace(",", ""))}
    m = EXTREME_RE.search(q)
    if m:
        return {"kind": "extreme", "sport": m.group("sport"), "games": m.group("games"), "field": m.group("field").lower(),
                "desc": m.group("sup").lower() in ("highest", "most", "largest", "greatest")}
    m = LOOKUP_RE.search(q)
    if m:
        medal = re.search(r"\b(gold|silver|bronze)\b", q, re.I)
        return {"kind": "lookup", "venue": m.group("venue").strip(), "date": m.group("date").strip(),
                "games": m.group("games"), "medal": medal.group(1).lower() if medal else None}
    return None


# =====================================================================
# 3) SELECT PAGES + COMPUTE, THEN BUILD A SMALL EVIDENCE TABLE
# =====================================================================
def _n(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _sport_pages(spec):
    qs, games = _n(spec["sport"]), spec["games"].lower()
    recs = [r for r in records() if r["games"].lower() == games]
    exact = [r for r in recs if _n(r["sport"]) == qs]
    return exact or [r for r in recs if qs in _n(r["sport"]) or _n(r["sport"]) in qs]


def _count_evidence(spec):
    recs = _sport_pages(spec)
    field = spec["field"]
    rows = [(r["event"], r[field], r["title"]) for r in recs if r.get(field) is not None]
    if not rows:
        return None
    hit = [x for x in rows if OPS[spec["op"]](x[1], spec["n"])]
    lines = [f"{spec['sport']} events at the {spec['games']} in the corpus ({len(recs)} pages, {field} listed for {len(rows)}):"]
    lines += [f"- {ev}: {v}" for ev, v, _ in rows]
    lines.append(f"Events with {field} {OP_TEXT[spec['op']]} {spec['n']}: {len(hit)}")
    return "\n".join(lines), [t for _, _, t in (hit or rows)][:6]


def _extreme_evidence(spec):
    field = spec["field"]
    rows = [(r["title"], r[field]) for r in _sport_pages(spec) if r.get(field) is not None]
    if not rows:
        return None
    rows.sort(key=lambda x: x[1], reverse=spec["desc"])
    lines = [f"{spec['sport']} events at the {spec['games']} ranked by {field} ({'highest' if spec['desc'] else 'lowest'} first):"]
    lines += [f"{i}. {t}: {v}" for i, (t, v) in enumerate(rows[:5], 1)]
    return "\n".join(lines), [t for t, _ in rows[:3]]


def _tok(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"}


def _dates(s):
    return {t for t in re.findall(r"\d+|[a-z]+", s.lower()) if t.isdigit() or t in _MONTHS}


def _venue_score(a, b):
    na, nb = _n(a), _n(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.95
    ta, tb = _tok(a), _tok(b)
    return len(ta & tb) / len(ta | tb)


def _lookup_evidence(spec):
    games = spec["games"]
    year_m = re.search(r"\b(1[89]\d\d|20\d\d)\b", (games or "") + " " + spec["date"])
    year = year_m.group(1) if year_m else None
    qd = _dates(spec["date"]) | ({year} if year else set())
    scored = []
    for r in records():
        vs = _venue_score(spec["venue"], r["venue"]) if r["venue"] else 0.0
        if vs < 0.6:
            continue
        gm = 1 if (not games or r["games"].lower() == games.lower()) and (not year or year in r["games"]) else 0
        rd = _dates(r["dates"])
        ds = len(qd & rd) / len(rd) if rd else 0.0
        scored.append((gm * 10 + (5 if vs >= 0.9 else vs * 3) + ds, r))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0]
    top = [r for s, r in scored[:2] if s >= best - 0.01]
    blocks = []
    for r in top:
        extra = "; ".join(f"{k}: {v[:90]}" for k, v in r["fields"].items() if k not in ("event", "games"))[:600]
        blocks.append(f"{r['title']}\n  {extra}")
    return "\n".join(blocks), [r["title"] for r in top]


def evidence(spec):
    """(evidence_text, page_titles) or None when the corpus has no matching pages."""
    return {"count": _count_evidence, "extreme": _extreme_evidence, "lookup": _lookup_evidence}[spec["kind"]](spec)


# =====================================================================
# 4) THE LLM WRITES THE ANSWER FROM THE EVIDENCE TABLE
# =====================================================================
def answer(q):
    spec = parse(q)
    if not spec:
        return None
    ev = evidence(spec)
    if ev is None:
        rp.log(f"  [table] {spec['kind']}: no matching event pages")
        return None
    text, titles = ev
    rp.trace.append(text)
    rp.cited.extend(t for t in titles if t not in rp.cited)
    rp.log(f"  [table] {spec['kind']} over parsed infobox fields -> evidence of ~{rp.est_tokens(text)} tokens")
    return rp.llm([
        {"role": "system", "content": TABLE_SYSTEM},
        {"role": "user", "content": f"Evidence:\n{text}\n\nQuestion: {q}\nAnswer:"},
    ], max_tokens=60)


# =====================================================================
# DEBUG CLI (no LLM needed)
# =====================================================================
def _shape(q):
    q = re.sub(r"\d[\d,\u2013\-]*", "<N>", q)
    return " ".join(q.split()[:8])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["build", "ask", "shapes"])
    ap.add_argument("question", nargs="?")
    a, _ = ap.parse_known_args()
    if a.cmd == "build":
        recs = records(force=True)
        print(f"{len(recs)} event pages | with competitors: {sum(r['competitors'] is not None for r in recs)} | "
              f"with venue: {sum(bool(r['venue']) for r in recs)} | with dates: {sum(bool(r['dates']) for r in recs)}")
        print("field names seen:", Counter(k for r in recs for k in r["fields"]).most_common(15))
        for r in recs[:3]:
            print(json.dumps({k: r[k] for k in ("title", "venue", "dates", "competitors", "nations", "fields")},
                             ensure_ascii=False)[:700])
    elif a.cmd == "ask":
        spec = parse(a.question)
        print("spec:", spec)
        ev = evidence(spec) if spec else None
        print(ev[0] if ev else "(no structured evidence)")
    elif a.cmd == "shapes":
        qs = [rp.pick(r, ("question", "query", "q")) for r in rp.read_jsonl(rp.EVAL_PATH)]
        for shape, n in Counter(_shape(q) for q in qs).most_common(15):
            print(f"{n:3d}  {shape}")
        print(f"\n{sum(parse(q) is not None for q in qs)}/{len(qs)} questions are handled by the table tool")
    else:
        print("Commands: build | ask \"question\" | shapes")
