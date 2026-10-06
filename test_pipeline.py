"""Fast unit tests: no Ollama, no Chroma, no TigerGraph needed (the LLM and retrievers are faked)."""
import json

import agentic as ag
import rag_pipeline as rp


def hit(i, title, text, chunk=0, score=0.9):
    return rp.Hit(f"d::{i}", "d", title, chunk, text, score)


def test_chunking_overlaps_and_keeps_every_sentence():
    sents = [f"Sentence number {i} talks about the Olympic Games in some detail." for i in range(60)]
    chunks = rp.chunk_doc({"doc_id": "d", "title": "T", "text": " ".join(sents)}, size=50, overlap=15)
    assert len(chunks) > 1
    joined = " ".join(c["text"] for c in chunks)
    assert all(s in joined for s in sents)
    last_of_first = rp.SENT_SPLIT.split(chunks[0]["text"])[-1]
    assert last_of_first in chunks[1]["text"]            # consecutive chunks overlap


def test_infobox_runs_are_split_into_fields():
    text = ("event: Men's 200 metres games: 2008 Summer venue: Beijing National Stadium dates: August 18 competitors: 39 "
            "nations: 31 gold: Usain Bolt silver: Churandy Martina bronze: Wallace Spearmon note: result later amended "
            "by the IOC after a disqualification ruling in the final heat of the competition")
    fields = rp.units(text)
    assert len(fields) > 5 and any(f.startswith("gold:") for f in fields)


def test_build_context_dedupes_overlap_and_records_sources():
    rp.cited.clear()
    a = hit(0, "Page A", "Alpha won gold. Beta won silver.")
    b = hit(1, "Page A", "Beta won silver. Gamma won bronze.", chunk=1)
    ctx = rp.build_context("who won", [a, b], budget=500)
    assert ctx.count("Beta won silver.") == 1
    assert rp.cited == ["Page A"]


def test_router_needs_no_llm_for_clear_cases():
    assert rp.route("Which city hosted the 2016 Summer Olympics?") == "single"
    assert rp.route("Compare the medals won by Japan and Brazil at the 2016 Games") == "multi"


def test_metrics():
    assert rp.correct("Usain Bolt of Jamaica", "Usain Bolt")
    assert not rp.correct("Not found in corpus", "Usain Bolt")
    assert rp.lenient_correct("Bolt, Usain", "Usain Bolt")            # reordered words
    assert rp.token_f1("Usain Bolt", "Usain Bolt") == 1.0


def test_parse_plan_survives_bad_planner_output(monkeypatch):
    monkeypatch.setattr(ag, "default_tool", lambda q: "vector")
    good = ag.parse_plan('```json\n[{"q": "Who hosted?", "tool": "page"}, {"q": "Where is #1?", "tool": "oops"}]\n```', "Q")
    assert [s["tool"] for s in good] == ["page", "vector"]            # unknown tool replaced
    assert ag.parse_plan("not json at all", "Original question?") == [{"q": "Original question?", "tool": "vector"}]


def test_agentic_two_hops_substitutes_answers_and_cites(monkeypatch):
    def fake_llm(messages, max_tokens=60):
        system, user = messages[0]["content"], messages[-1]["content"]
        if system.startswith("You plan"):
            return json.dumps([{"q": "Which city hosted the 2016 Games?", "tool": "vector"},
                               {"q": "In which country is #1?", "tool": "vector"}])
        if "In which country is Rio de Janeiro?" in user:
            return "Brazil"
        if "hosted" in user and "Question: Which city" in user:
            return "Rio de Janeiro"
        return "Brazil"                                                # synthesis call

    monkeypatch.setattr(rp, "llm", fake_llm)
    monkeypatch.setattr(rp, "rerank", lambda q, hits, n: hits[:n])
    monkeypatch.setitem(rp.current, "route", "multi")
    monkeypatch.setitem(ag.TOOLS, "vector", lambda q, k=4: [hit(1, "2016 Summer Olympics", "Rio de Janeiro hosted the Games.")])
    rp.steps.clear(); rp.cited.clear()
    assert ag.agentic_graphrag("In which country was the city that hosted the 2016 Games?") == "Brazil"
    assert "2016 Summer Olympics" in rp.cited
    assert any("hop 2: In which country is Rio de Janeiro?" in s for s in rp.steps)


def test_agentic_widens_search_once_when_a_hop_finds_nothing(monkeypatch):
    calls = []

    def fake_llm(messages, max_tokens=60):
        calls.append(1)
        return "INSUFFICIENT" if len(calls) == 1 else "Rio de Janeiro"

    monkeypatch.setattr(rp, "llm", fake_llm)
    monkeypatch.setattr(rp, "rerank", lambda q, hits, n: hits[:n])
    monkeypatch.setitem(rp.current, "route", "single")
    monkeypatch.setattr(ag, "default_tool", lambda q: "vector")
    monkeypatch.setattr(ag, "tool_vector", lambda q, k=4: [hit(1, "P", "Rio de Janeiro hosted the Games.")])
    monkeypatch.setattr(ag, "tool_graph", lambda q: [])
    monkeypatch.setattr(ag, "tool_page", lambda q, **kw: [])
    monkeypatch.setitem(ag.TOOLS, "vector", lambda q, k=4: [hit(1, "P", "Rio de Janeiro hosted the Games.")])
    assert ag.agentic_graphrag("Who hosted?") == "Rio de Janeiro"
    assert len(calls) == 2                                              # one failed attempt + one widened attempt


def test_orchestrator_escalates_then_falls_back_to_best_effort(monkeypatch):
    monkeypatch.setattr(rp, "route", lambda q: "single")
    monkeypatch.setitem(rp.PIPELINES, "rag", lambda q: "INSUFFICIENT")
    monkeypatch.setitem(rp.PIPELINES, "graphrag", lambda q: "Paris")
    r = rp.ask("Capital?")
    assert (r["pipeline"], r["answer"]) == ("graphrag", "Paris")

    def dead_end(q):
        rp.trace.append("[Page] some evidence")
        return "INSUFFICIENT"

    for name in ("rag", "graphrag", "agentic"):
        monkeypatch.setitem(rp.PIPELINES, name, dead_end)
    monkeypatch.setattr(rp, "llm", lambda messages, max_tokens=60: "best guess")
    r = rp.ask("Capital?")
    assert (r["pipeline"], r["answer"]) == ("best_effort", "best guess")


def test_safe_ask_never_raises(monkeypatch):
    def boom(q, force=None):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(rp, "ask", boom)
    r = rp.safe_ask("Q?")
    assert r["pipeline"] == "error" and r["answer"] == "Not found in corpus"


def test_predict_resumes_without_duplicates(tmp_path, monkeypatch):
    inp, out = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text("\n".join(json.dumps({"id": i, "question": f"Q{i}?"}) for i in range(3)), encoding="utf-8")
    monkeypatch.setattr(rp, "safe_ask", lambda q, force=None: {
        "answer": "A", "sources": [], "pipeline": "rag", "prompt_tokens": 1, "completion_tokens": 1, "path": []})
    rp.predict(str(inp), str(out), limit=2)
    rp.predict(str(inp), str(out), resume=True)
    qs = [json.loads(l)["question"] for l in out.read_text(encoding="utf-8").splitlines()]
    assert qs == ["Q0?", "Q1?", "Q2?"]
