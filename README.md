AI-Powered RAG Agent using LangGraph + Llama 3.1

A fully local Retrieval-Augmented Generation (RAG) pipeline built using LangGraph, Ollama (Llama 3.1), and Streamlit, enhanced with TruLens trace logging and automatic evaluation (BLEU, ROUGE, and BERTScore).

📘 Project Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) workflow for answering questions based on the content of a large document (e.g., a Renewable Energy PDF).

It leverages:

🧩 LangGraph – to define a multi-step reasoning pipeline (plan → retrieve → answer → reflect).

🦙 Llama 3.1 (via Ollama) – as the local LLM for answer generation and reflection.

📚 Chroma Vector Database – to store and retrieve text embeddings.

🔍 Hugging Face Sentence Transformers – for semantic embeddings.

📊 TruLens – for trace logging, explainability, and quality evaluation.

🌐 Streamlit UI – for an interactive Q&A interface.


User Query
   │
   ▼
[Plan Node] ─ Decide whether to retrieve context
   │
   ▼
[Retrieve Node] ─ Get relevant document chunks via Chroma DB
   │
   ▼
[Answer Node] ─ Use Llama 3.1 to generate context-grounded answer
   │
   ▼
[Reflect Node] ─ Evaluate accuracy & relevance of answer
   │
   ▼
[TruLens] ─ Log run data, feedback, and evaluation metrics

🧩 Key Components
Component	Purpose
LangGraph	Defines a graph-based RAG pipeline with nodes & state transitions.
Llama 3.1 (Ollama)	Generates answers and performs self-reflection locally.
Chroma Vector Store	Stores embeddings of the document for retrieval.
HuggingFace Embeddings	Creates embeddings (all-MiniLM-L6-v2) for semantic search.
TruLens	Logs LLM calls, tracks metrics, and evaluates quality of RAG responses.
BLEU / ROUGE / BERTScore	Evaluates similarity between generated and reference answers.
Streamlit	Provides an interactive UI for querying the model.

🧰 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/langgraph-llama-rag.git
cd langgraph-llama-rag

2️⃣ Create and Activate Virtual Environment
python -m venv hackathon
hackathon\Scripts\activate        # Windows
# OR
source hackathon/bin/activate     # Linux/Mac

3️⃣ Install Dependencies
pip install -r requirements.txt


Example requirements.txt

langchain-community>=0.2.0
langgraph>=0.0.10
trulens-eval>=1.0.0
trulens-apps-langchain>=1.0.0
streamlit
chromadb
sentence-transformers
evaluate
ollama

4️⃣ Ensure Ollama and Llama 3.1 Are Installed
ollama pull llama3.1

5️⃣ Place Your PDF in the Project Folder

For example:

Renewable_Energy.pdf

▶️ Run the App

Start the Streamlit interface:

streamlit run rag_agent_llama_streamlit.py


Then open your browser at:

http://localhost:8501

💡 Example Interaction

Question:

What is Nuclear Energy?

Generated Answer (by Llama 3.1):

Nuclear energy is the energy stored in the nucleus of an atom that holds the nucleus together. The nucleus of a uranium atom is an example.

Reflection:

Relevant — The answer provides a correct and concise definition aligned with the question.

Evaluation Scores:

Metric	Score
BLEU	0.92
ROUGE-L	0.88
BERTScore (F1)	0.95


📊 TruLens Logging and Dashboard

TruLens automatically logs each query-answer pair.
To view the dashboard:

from trulens_eval import Tru
tru = Tru()
tru.run_dashboard()


Then open:

http://localhost:8501/trulens

🧱 Folder Structure
├── rag_agent_llama_streamlit.py   # Main app file
├── Renewable_Energy.pdf           # Knowledge base
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── chroma_store_llama/            # Local vector store
└── trulens_data/                  # Logs and traces
