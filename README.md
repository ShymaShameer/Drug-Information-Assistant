# 💊 Drug Information Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application built with **LlamaIndex, Groq LLM, and Streamlit**. It helps users explore official FDA drug-labeling documents and investigate how prescribing information has evolved over time.

The assistant uses **United States Prescribing Information (USPI) documents from 2013 to the present** for selected antidiabetic medications, allowing users to ask drug-specific questions, retrieve relevant regulatory passages, and compare labeling information across different years.

🔗 **Live Web Application:** [ Drug Information Assistant](http://localhost:8501/)

---

## 📚 Data Scope

The knowledge base contains historical and current U.S. prescribing information for seven antidiabetic medications covering two drug classes:

- **SGLT2 inhibitors**
- **GLP-1 receptor agonists**

| Data characteristic | Description |
|---|---|
| Regulatory source | U.S. Food and Drug Administration (FDA) |
| Document type | United States Prescribing Information (USPI) |
| Time coverage | 2013 to present |
| Drug coverage | Seven selected antidiabetic medications |
| Main focus | Drug safety information and changes in regulatory labeling |

The documents cover clinically relevant labeling sections, including:

- Contraindications
- Warnings and precautions
- Adverse reactions
- Drug interactions
- Use in specific populations

**Important:** This application focuses on information contained in official prescribing information. It does not independently analyze individual FAERS case reports or establish causal relationships between medications and adverse events.

---

## 🌟 Key Features

### 1. Drug-Specific Document Retrieval
Select a medication to explore its corresponding prescribing information and retrieve relevant passages from the available regulatory documents.

### 2. Historical Year Selection
Filter the knowledge base by year to investigate what the prescribing information stated at a particular point in time.

### 3. Historical Label Comparison
Compare prescribing information across selected years to explore how safety-related regulatory language has changed over time.

### 4. RAG-Based Question Answering
Ask natural-language questions about drug safety and receive context-grounded answers generated using retrieved passages from the source documents.

### 5. HyDE and Re-Ranking Architecture
Uses **Hypothetical Document Embeddings (HyDE)** to expand short or vague queries, followed by a **SentenceTransformer reranker** (`BAAI/bge-reranker-large`) to improve the relevance of retrieved passages.

### 6. Source Attribution and Traceability
Displays retrieved document passages and their available source information, such as file names, page numbers, and similarity scores, allowing users to inspect the underlying regulatory text.

### 7. Configurable RAG Parameters
Provides controls for retrieval depth (`similarity_top_k`) and system prompt selection to customize retrieval and response behavior.

### 8. Interactive Streamlit Interface
Includes a chat interface, drug and year selection, historical comparison functionality, and a **New Chat** option to reset the conversation.

---

## 🏗️ Project Architecture

The application follows a modular RAG architecture:

1. **Document ingestion:** Historical and current USPI documents are loaded and prepared for retrieval.
2. **Document processing:** Source documents are split into retrievable text chunks while retaining relevant document metadata.
3. **Embedding generation:** Text chunks are converted into vector representations.
4. **Vector retrieval:** Relevant passages are retrieved based on the user's query and selected drug/year filters.
5. **HyDE and re-ranking:** Query expansion and reranking help improve contextual relevance.
6. **LLM response generation:** The retrieved context is passed to the Groq-powered LLM to generate an answer.
7. **Source verification:** Retrieved passages and their available metadata are presented for reference.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| RAG framework | LlamaIndex |
| LLM provider | Groq |
| Embeddings | Hugging Face Sentence Transformers |
| Re-ranking | `BAAI/bge-reranker-large` |
| Vector retrieval | LlamaIndex vector index |
| User interface | Streamlit |
| Environment management | Conda / pip |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ShymaShameer/Drug-Information-Assistant.git
cd Drug-Information-Assistant
```

### 2. Create and activate an environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate your-environment-name
```

Alternatively, using pip:

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit API keys or other sensitive credentials to GitHub.

### 4. Run the application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Drug-Information-Assistant/
├── .vscode/                 # VS Code workspace settings
├── app.py                   # Streamlit application interface
├── src/
│   ├── engine.py            # Core RAG pipeline and retrieval logic
│   ├── model_loader.py      # LLM and embedding model initialization
│   └── config.py            # Application configuration and RAG parameters
├── local_storage/           # Persistent vector index and storage (Git ignored)
├── .env                     # API keys and environment variables (Git ignored)
├── .gitignore               # Git exclusion rules
├── environment.yml          # Conda environment definition
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

*Note: Adjust the directory structure above if your current repository contains additional files or folders.*

---

## 🎯 Project Objective

The goal of this project is to make historical regulatory drug-labeling information easier to explore and compare.

By combining document retrieval, large language models, and year-based filtering, the assistant provides an interactive way to investigate how prescribing information has evolved and to support historical verification of drug safety information.

The extended application also demonstrates how RAG can be used beyond conventional question answering to support **longitudinal exploration of regulatory documents**.

---

## ⚠️ Limitations

- Responses depend on the availability, quality, and completeness of the indexed prescribing information.
- Generated answers should always be checked against the original FDA labeling documents.
- The application is an information-retrieval and research-support tool, not a substitute for professional medical advice or regulatory assessment.
- The presence or absence of information in a label does not independently establish whether an adverse event is causally related to a medication.

---

## 👩‍💻 Author

**Shyma Shameer**

GitHub: [@ShymaShameer](https://github.com/ShymaShameer)