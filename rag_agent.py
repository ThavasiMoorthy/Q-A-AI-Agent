# ========================================================
# 🚫 Disable OpenTelemetry BEFORE ANYTHING ELSE
# ========================================================
import os
os.environ["TRULENS_OTEL_ENABLED"] = "0"
os.environ["OTEL_PYTHON_DISABLED"] = "true"
os.environ["OTEL_TRACING_ENABLED"] = "false"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"

"""
rag_agent_llama_streamlit.py
Author: [Your Name]
Description: Fully local LangGraph + RAG Agent using Llama 3.1 (via Ollama)
with Streamlit UI, TruLens trace logging, and BLEU/ROUGE/BERTScore evaluation.
"""

# ===============================
# Imports
# ===============================
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from trulens_eval import Tru, TruChain
from evaluate import load
import streamlit as st


# ===============================
# PDF Path
# ===============================
PDF_PATH = "Renewable_Energy.pdf"


# ===============================
# Graph State Class
# ===============================
class GraphState:
    def __init__(self, query=None, docs=None, answer=None, reflection=None, retrieve_needed=True):
        self.query = query
        self.docs = docs
        self.answer = answer
        self.reflection = reflection
        self.retrieve_needed = retrieve_needed


# ===============================
# Load & Prepare Knowledge Base
# ===============================
st.write("📥 Loading and preparing the knowledge base...")

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
st.success(f"✅ Loaded and split {len(chunks)} chunks from PDF.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_store_llama")
st.success("✅ Chroma vector store created using HuggingFace embeddings.")


# ===============================
# Define LangGraph Nodes
# ===============================
def plan_node(state: GraphState):
    q = state.query.lower()
    state.retrieve_needed = any(word in q for word in ["what", "how", "explain", "define", "describe"])
    st.info(f"🧭 Plan Node → Retrieval needed: {state.retrieve_needed}")
    return state


def retrieve_node(state: GraphState):
    if state.retrieve_needed:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(state.query)
        state.docs = [d.page_content for d in docs]
        st.info(f"🔍 Retrieve Node → Retrieved {len(state.docs)} chunks.")
    else:
        state.docs = []
        st.info("🔍 Retrieve Node → Skipped retrieval.")
    return state


def answer_node(state: GraphState):
    llm = Ollama(model="llama3.1")
    context = "\n\n".join(state.docs) if state.docs else ""
    prompt = f"""
You are a renewable energy expert.
Use the context below to answer the user's question clearly and concisely.

Context:
{context}

Question:
{state.query}

If you don't find the answer in the context, say:
"I couldn’t find relevant information in the provided documents."
"""
    state.answer = llm.invoke(prompt)
    st.success("💬 Answer Node → Answer generated successfully.")
    return state


def reflect_node(state: GraphState):
    llm = Ollama(model="llama3.1")
    reflection_prompt = f"""
Evaluate if this answer correctly addresses the question.
Question: {state.query}
Answer: {state.answer}
Reply with 'Relevant' or 'Not relevant' and explain briefly.
"""
    state.reflection = llm.invoke(reflection_prompt)
    st.success("✅ Reflect Node → Reflection completed.")
    return state


# ===============================
# Build LangGraph Workflow
# ===============================
graph = StateGraph(GraphState)
graph.add_node("plan", plan_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("reflect", reflect_node)

graph.add_edge("plan", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", "reflect")
graph.add_edge("reflect", END)
graph.set_entry_point("plan")

app = graph.compile()


# ===============================
# TruLens Integration (TruChain)
# ===============================
tru = Tru()
tru_chain = TruChain(app=app, chain_name="LangGraph_RAG_Pipeline")


# ===============================
# Wrap LangGraph with TruLens
# ===============================
def run_agent(query):
    """
    Run the LangGraph pipeline with TruLens tracking.
    """
    with tru_chain as recording:
        state = GraphState(query=query)
        result = app.invoke(state)
        # No need for record_metadata() – TruChain auto-logs everything.
    return result


# ===============================
# Streamlit Interface
# ===============================
st.title("⚡ AI Q&A Agent using LangGraph + Llama 3.1")
st.caption("Ask any question based on the Renewable Energy document!")

user_question = st.text_input("❓ Enter your question:")

if st.button("🚀 Run Agent"):
    if not user_question.strip():
        st.warning("Please enter a question before running the agent.")
    else:
        with st.spinner("Processing your question... ⏳"):
            st.info("📊 TruLens trace logging started...")
            final_state = run_agent(user_question)
            st.info("📈 TruLens trace logged successfully.")

        # ===============================
        # Display Results
        # ===============================
        st.subheader("🧠 Final Answer:")
        st.write(final_state.answer)

        st.subheader("🪞 Reflection:")
        st.write(final_state.reflection)

        # ===============================
        # Evaluation Metrics
        # ===============================
        llm = Ollama(model="llama3.1")
        reference_prompt = f"Provide a short textbook-style answer for: {user_question}"
        reference = [llm.invoke(reference_prompt)]
        prediction = [final_state.answer]

        bleu = load("bleu")
        rouge = load("rouge")
        bert = load("bertscore")

        bleu_score = bleu.compute(predictions=prediction, references=reference)
        rouge_score = rouge.compute(predictions=prediction, references=reference)
        bert_score = bert.compute(predictions=prediction, references=reference, lang="en")

        st.subheader("📊 Evaluation Metrics:")
        st.write(f"**BLEU:** {bleu_score['bleu']:.4f}")
        st.write(f"**ROUGE-L:** {rouge_score['rougeL']:.4f}")
        st.write(f"**BERTScore (F1):** {sum(bert_score['f1']) / len(bert_score['f1']):.4f}")

        st.success("✅ Process completed successfully!")
