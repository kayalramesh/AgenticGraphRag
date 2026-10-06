"""
TigerGraph GraphRAG for the Olympics corpus
===========================================
Graph model (all built from the chunks/embeddings you already created, no extra LLM calls):
 
    (Chunk {chunk_id, doc_id, title, chunk_idx, body, embedding[384]})  -[MENTIONS]-  (Entity {name})
 
Retrieval = ONE installed GSQL query (graphrag_retrieve):
    1. vectorSearch on Chunk.embedding               -> top-k seed chunks      (the "RAG" part)
    2. seed chunks -> MENTIONS -> Entity (+ entities named in the question)    (graph hop 1)
    3. Entity -> MENTIONS -> other Chunks             -> candidate chunks      (graph hop 2)
    4. vectorSearch restricted to candidate_set       -> best related chunks   (hybrid graph+vector re-rank)
 
Setup: put your credentials in a file named .env next to these scripts (loaded automatically; never commit it):
    TG_HOST=https://<your-savanna-workspace-host>
    TG_SECRET=<gsql secret>            # OR use TG_USERNAME and TG_PASSWORD instead
    TG_GRAPHNAME=OlympicsGraphRAG      # optional, this is the default (same name the TigerGraph MCP server reads)
Or set them in PowerShell for the current terminal only:
    uv pip install pyTigerGraph        # (uv venvs have no pip - plain `pip` installs into your global Python)
    $env:TG_HOST     = "https://<your-savanna-workspace-host>"   # copy from the Savanna workspace "Connect" panel
    $env:TG_SECRET   = "<gsql secret>"                           # preferred on Savanna; OR set the two lines below
    $env:TG_USERNAME = "tigergraph"
    $env:TG_PASSWORD = "<password>"
    $env:TG_GRAPH    = "OlympicsGraphRAG"                        # optional, this is the default
 
Run in this order (each step is safe to re-run):
    python tg_graphrag.py check      # connection test
    python tg_graphrag.py schema     # vertex/edge types + vector attribute + graph + loading jobs
    python tg_graphrag.py load       # export chunks/embeddings from Chroma and bulk-load (no re-embedding)
    python tg_graphrag.py install    # create + install the GSQL query
    python tg_graphrag.py test "Which athletes won medals at both the 2008 and 2012 Summer Olympics?"
"""
import argparse
import os
import time
import urllib.parse
from pathlib import Path
 
import pyTigerGraph as tg
from tqdm import tqdm
 
import rag_pipeline as rp
 
GRAPH = os.environ.get("TG_GRAPHNAME") or os.environ.get("TG_GRAPH", "OlympicsGraphRAG")   # TG_GRAPHNAME is what tigergraph-mcp reads
EMB_DIM = 384                        # bge-small-en-v1.5
MIN_ENT_FREQ = 2                     # entities seen in only 1 chunk cannot connect anything
TMP_DIR = Path("./tg_load_tmp")
 
 
# =====================================================================
# CONNECTION
# =====================================================================
_conn = None
 
 
def connect():
    global _conn
    if _conn is None:
        host = os.environ["TG_HOST"].rstrip("/")
        secret = os.environ.get("TG_SECRET")
        kw = dict(host=host, graphname=GRAPH)
        port = os.environ.get("TG_PORT") or ("443" if host.startswith("https") else None)
        if port:                                          # Savanna/cloud: REST and GSQL both on 443
            kw["restppPort"] = port
            kw["gsPort"] = port
        if "<" in host or ">" in host:
            raise SystemExit(f"TG_HOST still contains a placeholder: {host!r}. Put your real Savanna host in .env")
        if secret:
            kw["gsqlSecret"] = secret
        else:
            kw["username"] = os.environ.get("TG_USERNAME", "tigergraph")
            kw["password"] = os.environ["TG_PASSWORD"]
        conn = tg.TigerGraphConnection(**kw)
        if secret:
            try:
                conn.getToken(secret)
            except Exception as e:                        # some versions fetch the token themselves
                print(f"[tg] getToken warning: {str(e)[:120]}")
        _conn = conn
    return _conn
 
 
# =====================================================================
# SCHEMA  (global vertex/edge types + vector attribute, then the graph)
# =====================================================================
# Each statement runs on its own, so an "already exists" message on a re-run never blocks the later steps.
SCHEMA_STEPS = [
    "USE GLOBAL\nCREATE VERTEX Chunk (chunk_id STRING PRIMARY KEY, doc_id STRING, title STRING, chunk_idx INT, body STRING)",
    "USE GLOBAL\nCREATE VERTEX Entity (name STRING PRIMARY KEY)",
    "USE GLOBAL\nCREATE UNDIRECTED EDGE MENTIONS (FROM Chunk, TO Entity)",
    "USE GLOBAL\nCREATE GLOBAL SCHEMA_CHANGE JOB add_chunk_vec {\n"
    f'  ALTER VERTEX Chunk ADD VECTOR ATTRIBUTE embedding(DIMENSION={EMB_DIM}, METRIC="COSINE");\n'
    "}\nRUN GLOBAL SCHEMA_CHANGE JOB add_chunk_vec",
    f"USE GLOBAL\nCREATE GRAPH {GRAPH}(Chunk, Entity, MENTIONS)",
]
 
# '|' is the field separator because the embedding itself contains commas
JOB_STEPS = [
    f"USE GRAPH {GRAPH}\nCREATE LOADING JOB load_chunks FOR GRAPH {GRAPH} {{\n"
    "  DEFINE FILENAME f;\n"
    '  LOAD f TO VERTEX Chunk VALUES ($0, $1, $2, $3, $4) USING SEPARATOR="|", HEADER="false";\n'
    '  LOAD f TO VECTOR ATTRIBUTE embedding ON VERTEX Chunk VALUES ($0, SPLIT($5, ",")) USING SEPARATOR="|", HEADER="false";\n'
    "}",
    f"USE GRAPH {GRAPH}\nCREATE LOADING JOB load_entities FOR GRAPH {GRAPH} {{\n"
    "  DEFINE FILENAME f;\n"
    '  LOAD f TO VERTEX Entity VALUES ($0) USING SEPARATOR="|", HEADER="false";\n'
    "}",
    f"USE GRAPH {GRAPH}\nCREATE LOADING JOB load_mentions FOR GRAPH {GRAPH} {{\n"
    "  DEFINE FILENAME f;\n"
    '  LOAD f TO EDGE MENTIONS VALUES ($0, $1) USING SEPARATOR="|", HEADER="false";\n'
    "}",
]
 
QUERY_GSQL = f"""
USE GRAPH {GRAPH}
CREATE OR REPLACE QUERY graphrag_retrieve(LIST<FLOAT> query_vector, SET<STRING> q_names,
                                          INT k_seed = 4, INT k_extra = 4, INT entity_cap = {rp.HUB_CAP}) SYNTAX v3 {{
  MapAccum<Vertex, FLOAT> @@d_seed;
  MapAccum<Vertex, FLOAT> @@d_extra;
  MapAccum<STRING, FLOAT> @@seed_dist;
  MapAccum<STRING, FLOAT> @@extra_dist;
 
  // 1) vector search for seed chunks
  seed_hits = vectorSearch({{Chunk.embedding}}, query_vector, k_seed, {{distance_map: @@d_seed}});
 
  // 2) entities in the seed chunks + entities named in the question (skip hub entities)
  seed_ents = SELECT e FROM (s:seed_hits)-[:MENTIONS]-(e:Entity)
              WHERE e.outdegree("MENTIONS") <= entity_cap;
  q_set = SELECT e FROM (e:Entity) WHERE e.name IN q_names;
  anchors = seed_ents UNION q_set;
 
  // 3) graph hop: other chunks that mention those entities
  cand = SELECT c FROM (e:anchors)-[:MENTIONS]-(c:Chunk)
         WHERE e.outdegree("MENTIONS") <= entity_cap;
  cand = cand MINUS seed_hits;
 
  // 4) hybrid step: vector re-rank restricted to the graph candidates
  extra_hits = vectorSearch({{Chunk.embedding}}, query_vector, k_extra, {{candidate_set: cand, distance_map: @@d_extra}});
 
  t1 = SELECT s FROM (s:seed_hits) ACCUM @@seed_dist += (s.chunk_id -> @@d_seed.get(s));
  t2 = SELECT x FROM (x:extra_hits) ACCUM @@extra_dist += (x.chunk_id -> @@d_extra.get(x));
 
  PRINT seed_hits;
  PRINT extra_hits;
  PRINT @@seed_dist;
  PRINT @@extra_dist;
}}
INSTALL QUERY graphrag_retrieve
"""
 
 
def _scope(conn, graph=None):
    """pyTigerGraph 2.x: switch between global scope (schema work) and a graph."""
    if graph is None and hasattr(conn, "useGlobal"):
        conn.useGlobal()
    elif graph is not None and hasattr(conn, "useGraph"):
        conn.useGraph(graph)
 
 
def _run_steps(conn, steps):
    for i, stmt in enumerate(steps, 1):
        head = stmt.splitlines()[1][:80]
        try:
            out = conn.gsql(stmt)
            print(f"[{i}/{len(steps)}] {head}\n     -> {str(out).strip()[:300]}")
        except Exception as e:
            print(f"[{i}/{len(steps)}] {head}\n     !! {str(e).strip()[:300]}")
 
 
def cmd_check():
    conn = connect()
    print("echo:", conn.echo())
    try:
        _scope(conn)
        print("global schema / graphs:\n", str(conn.gsql("LS"))[:1500])
    except Exception as e:
        print(f"(could not list schema: {str(e)[:150]})")
 
 
def cmd_schema():
    conn = connect()
    _scope(conn)
    _run_steps(conn, SCHEMA_STEPS)
    _scope(conn, GRAPH)
    _run_steps(conn, JOB_STEPS)
    print("\nSchema step finished. Lines starting with '!!' that say 'already exists' are fine on a re-run.")
 
 
# =====================================================================
# EXPORT FROM CHROMA + BULK LOAD (reuses the embeddings you already computed)
# =====================================================================
def _safe(s):
    return str(s).replace("|", " / ").replace("\n", " ").replace("\r", " ")
 
 
def _run_file(job, lines, tag):
    TMP_DIR.mkdir(exist_ok=True)
    path = TMP_DIR / f"{job}_{tag}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    res = connect().runLoadingJobWithFile(str(path), "f", job, "|")
    path.unlink(missing_ok=True)
    return res
 
 
def cmd_load(chunk_batch=1500, line_batch=100_000):
    col = rp.collection()
    total = col.count()
    idx = rp.entity_index()
    keep = {e for e, ids in idx.items() if MIN_ENT_FREQ <= len(ids) <= rp.HUB_CAP}
    n_edges = sum(len(idx[e]) for e in keep)
    print(f"[load] {total} chunks, {len(keep)} entities, {n_edges} MENTIONS edges")
 
    for off in tqdm(range(0, total, chunk_batch), desc="chunks+vectors"):
        got = col.get(limit=chunk_batch, offset=off, include=["documents", "metadatas", "embeddings"])
        lines = []
        for cid, doc, m, emb in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"]):
            lines.append("|".join([_safe(cid), _safe(m["doc_id"]), _safe(m["title"]), str(m["chunk_idx"]),
                                   _safe(doc), ",".join(f"{x:.5f}" for x in emb)]))
        res = _run_file("load_chunks", lines, off)
        if off == 0:
            print("[load] first batch result:", str(res)[:400])
 
    ent_lines = [_safe(e) for e in keep]
    for i in range(0, len(ent_lines), line_batch):
        _run_file("load_entities", ent_lines[i:i + line_batch], i)
 
    buf, part = [], 0
    for e in tqdm(keep, desc="mentions"):
        for cid in idx[e]:
            buf.append(f"{_safe(cid)}|{_safe(e)}")
        if len(buf) >= line_batch:
            _run_file("load_mentions", buf, part)
            buf, part = [], part + 1
    if buf:
        _run_file("load_mentions", buf, part)
 
    conn = connect()
    print("[load] Chunk:", conn.getVertexCount("Chunk"), "| Entity:", conn.getVertexCount("Entity"),
          "| MENTIONS:", conn.getEdgeCount("MENTIONS"))
    print("[load] The vector index builds in the background - wait ~1-2 minutes before `test`.")
 
 
def cmd_install():
    print(connect().gsql(QUERY_GSQL))
 
 
# =====================================================================
# RETRIEVAL used by the orchestrator (returns rag_pipeline.Hit objects)
# =====================================================================
def _to_hits(vertices, dist):
    hits = []
    for v in vertices or []:
        a = v.get("attributes", {})
        cid = a.get("chunk_id", v.get("v_id"))
        d = float(dist.get(cid, 0.5))                      # cosine distance -> similarity
        hits.append(rp.Hit(cid, a.get("doc_id", ""), a.get("title", ""), int(a.get("chunk_idx", 0)),
                           a.get("body", ""), 1.0 - d))
    return hits
 
 
def tg_retrieve(question, k_seed=4, k_extra=4):
    qv = rp.embed_query(question)
    idx = rp.entity_index()
    names = [e for e in rp.extract_entities(question) if e in idx and MIN_ENT_FREQ <= len(idx[e]) <= rp.HUB_CAP][:8]
    names.append("__none__")                                # SET params must not be empty
    conn = connect()
    # pyTigerGraph 2.x: dict params -> POST (JSON body, no URL-length limit); str params -> GET
    body = {"query_vector": [round(float(x), 5) for x in qv], "q_names": names, "k_seed": k_seed, "k_extra": k_extra}
    try:
        res = conn.runInstalledQuery("graphrag_retrieve", body, timeout=60000)
    except Exception:                                       # fall back to a GET query string
        qs = "&".join(
            [f"query_vector={x:.5f}" for x in qv]
            + [f"q_names={urllib.parse.quote(n)}" for n in names]
            + [f"k_seed={k_seed}", f"k_extra={k_extra}"]
        )
        res = conn.runInstalledQuery("graphrag_retrieve", qs, timeout=60000)
    out = {}
    for item in res:
        out.update(item)
    seeds = _to_hits(out.get("seed_hits"), out.get("@@seed_dist", {}))
    extra = _to_hits(out.get("extra_hits"), out.get("@@extra_dist", {}))
    return sorted(seeds + extra, key=lambda h: -h.score)
 
 
def cmd_test(question):
    t0 = time.time()
    hits = tg_retrieve(question)
    print(f"[tg] {len(hits)} chunks in {time.time() - t0:.2f}s")
    if not hits:
        print("No hits. The vector index may still be building, or the query is not installed (run `install`).")
        return
    for h in hits:
        print(f"  {h.score:.3f}  [{h.title}] {h.text[:110]}...")
    print("\n[answer via orchestrator, forced to graphrag]")
    print(rp.ask(question, force="graphrag"))
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["check", "schema", "load", "install", "test"])
    ap.add_argument("question", nargs="?")
    a, _ = ap.parse_known_args()
    if a.cmd == "check":
        cmd_check()
    elif a.cmd == "schema":
        cmd_schema()
    elif a.cmd == "load":
        cmd_load()
    elif a.cmd == "install":
        cmd_install()
    elif a.cmd == "test":
        cmd_test(a.question)
    else:
        print("Commands: check | schema | load | install | test \"question\"")
 