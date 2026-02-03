## RAG Web App – Simplified Document Chat System

This is a modern Retrieval-Augmented Generation (RAG) web app built with **Flask**, **LangChain**, and **Groq**. It allows users to:

- Upload PDF or text documents
- Ask questions about the uploaded documents
- Chat with the content using a powerful LLM (`llama3-8b-8192`)
- Enjoy a clean and user-friendly UI

---

## Features

- File Upload (PDF, TXT)
- Embedding with HuggingFace (`all-MiniLM-L6-v2`)
- LLM: Groq + Llama 3 (8B)
- Semantic search using FAISS
- Chunking with LangChain
- Clean UI (inspired by modern SaaS design)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yusufadepoju1/rag-app.git
cd rag-app
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Create a `.env` file

```
GROQ_API_KEY=your-groq-api-key-here
```

### 5. Run the app

```bash
python app.py
```

Then go to: (http://127.0.0.1:5000)

---

## Project Structure

```
rag_app/
├── app.py
├── uploads/             
├── templates/
│   ├── upload.html        
│   └── chat.html
|   ├── index.html          
├── .env                    
├── requirements.txt
└── README.md
```

---

## Requirements

```
Flask
langchain
langchain-groq
langchain-community
python-dotenv
faiss-cpu
sentence-transformers
```

> These are included in the `requirements.txt`

---





