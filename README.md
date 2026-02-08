# 🤖 IntelliRAG-System: Intelligent Document Assistant

An advanced **Retrieval-Augmented Generation (RAG)** system that allows you to have a conversation with your PDF documents. This project integrates high-performance vector search with state-of-the-art Open-Source LLMs.



## 🌟 Key Features
- **High-Speed Inference**: Powered by **Groq Cloud** using the LPU™ Inference Engine for near-instant responses.
- **Advanced Retrieval**: Utilizes **ChromaDB** for efficient vector similarity search.
- **Open-Source Excellence**: Features **Llama 3.3-70b** for sophisticated reasoning and analysis.
- **Smart Citations**: Automatically tracks and displays source references and page numbers from the PDF.
- **Professional UI**: A sleek, user-friendly chat interface built with **Streamlit**.

## 🛠️ Tech Stack
- **LLM Engine:** Llama 3.3-70b (via Groq API)
- **Vector Database:** ChromaDB
- **Embedding Model:** HuggingFace `all-MiniLM-L6-v2`
- **Framework:** LangChain & Python
- **Frontend:** Streamlit

## 🏗️ System Architecture
1. **Ingestion**: Documents are split into 1,000-character chunks with 200-character overlap.
2. **Embedding**: Text chunks are converted into 384-dimensional vectors.
3. **Storage**: Vectors are stored in a local ChromaDB instance.
4. **Retrieval**: User queries trigger a similarity search to find the most relevant context.
5. **Generation**: The retrieved context and user query are sent to Llama 3.3 to generate a precise answer.



## ⚙️ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Punyisa-m/IntelliRAG-System.git
cd IntelliRAG-System

```

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

### 3. Setup Environment Variables

Create a `.env` file in the root directory and add your API Key:

```text
GROQ_API_KEY=your_groq_api_key_here

```

### 4. Ingest Documents

Place your PDF files in the `data/` folder and run:

```bash
python src/ingest.py

```

### 5. Launch the Chat App

```bash
streamlit run src/app.py

```
