"""Demo UI:  streamlit run app.py   (shows the 3 pipelines side by side + the agent's investigation path)"""
import streamlit as st
import agentic_graphrag as ag

st.set_page_config(page_title="Olympics: RAG vs GraphRAG vs Agentic GraphRAG", layout="wide")
st.title("Olympics investigation: RAG vs GraphRAG vs Agentic GraphRAG")
q = st.text_input("Question", "Which athletes won medals at both the 2008 and 2012 Summer Olympics?")
if st.button("Ask all three") and q:
    cols = st.columns(3)
    for col, name in zip(cols, ag.PIPES):
        with col, st.spinner(name):
            r = ag.ask_any(q, name)
            st.subheader(name)
            st.write(r["answer"])
            st.caption(f"{r['seconds']}s | {r['llm_calls']} LLM calls | {r['prompt_tokens'] + r['completion_tokens']} tokens")
            if name == "agentic":
                with st.expander("Investigation path", expanded=True):
                    for s in r["steps"]:
                        st.markdown(f"**{s['step']}. {s['action']}** `{s['args']}`  \n{s['observation'][:250]}")
                    st.caption(f"stop: {r['stop_reason']}")
                with st.expander("Citations"):
                    for c in r["citations"]:
                        st.write(f"[{c['n']}] {c['title']} — `{c['chunk']}`")
