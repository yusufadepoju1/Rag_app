# from flask import Flask, render_template_string, request, jsonify
# import os
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader, PyPDFLoader
# import os
# from dotenv import load_dotenv
# from flask import render_template 


# load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Initialize Groq LLM with LangChain
# #groq_api_key = "your-groq-api-key-here"
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# # Embedding model
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = None

# def load_document(file_path):
#     _, ext = os.path.splitext(file_path)
#     if ext == '.txt':
#         loader = TextLoader(file_path)
#     elif ext == '.pdf':
#         loader = PyPDFLoader(file_path)
#     else:
#         return []
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return text_splitter.split_documents(docs)


 

# @app.route('/')
# def home():
#     return render_template('index.html')




# @app.route('/upload', methods=['POST'])
# def upload_file():
#     global vectorstore
#     if 'file' not in request.files:
#         return "No file part", 400
#     file = request.files['file']
#     if file.filename == '':
#         return "No selected file", 400
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)
#     docs = load_document(file_path)
#     if not docs:
#         return "Unsupported file type", 400
#     if vectorstore is None:
#         vectorstore = FAISS.from_documents(docs, embeddings)
#     else:
#         vectorstore.add_documents(docs)
#     return f"File '{file.filename}' uploaded and processed!"

# @app.route('/chat', methods=['POST'])
# def chat():
#     global vectorstore
#     data = request.get_json()
#     question = data.get("question", "").strip()
#     if not vectorstore:
#         return jsonify({"answer": "Please upload a document first!"})
#     try:
#         qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
#         result = qa_chain.invoke({"query": question})
#         return jsonify({"answer": result["result"]})
#     except Exception as e:
#         return jsonify({"answer": f"Error: {str(e)}"})

# if __name__ == '__main__':
#     app.run(debug=True)













from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = None

def load_document(file_path):
    _, ext = os.path.splitext(file_path)
    if ext == '.txt':
        loader = TextLoader(file_path)
    elif ext == '.pdf':
        loader = PyPDFLoader(file_path)
    else:
        return []
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

@app.route('/')
@app.route('/upload')
def upload_page():
    files = os.listdir(UPLOAD_FOLDER)
    return render_template("upload.html", files=files)

@app.route('/chat')
def chat_page():
    return render_template("chat.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    global vectorstore
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    docs = load_document(file_path)
    if not docs:
        return "Unsupported file type", 400

    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        vectorstore.add_documents(docs)

    return f"File '{file.filename}' uploaded and processed!"

@app.route('/chat', methods=['POST'])
def chat():
    global vectorstore
    data = request.get_json()
    question = data.get("question", "").strip()
    if not vectorstore:
        return jsonify({"answer": "Please upload a document first!"})
    try:
        qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
        result = qa.invoke({"query": question})
        return jsonify({"answer": result["result"]})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
