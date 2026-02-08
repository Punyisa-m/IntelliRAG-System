import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# API Key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 1. Setting UI
st.set_page_config(page_title="IntelliRAG-System", layout="wide")
st.title("🤖 IntelliRAG-System")
st.markdown("Intelligent Retrieval-Augmented Generation Platform for Document Analysis")
st.markdown("---")

# 2. funtion load database
@st.cache_resource
def load_db():
    #  Embedding HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

db = load_db()

# 3. Setting Groq Client
if not GROQ_API_KEY:
    st.error("Not found GROQ_API_KEY in .env")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Chat Interface)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get input
if user_input := st.chat_input("Ask about Apple 10-K:"):
    # Save Q from User
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG Process
    with st.chat_message("assistant"):
        with st.spinner("Prosessing..."):
            try:
                # 1. Search docs from ChromaDB
                docs = db.similarity_search(user_input, k=3)
                context_text = "\n\n".join([d.page_content for d in docs])

                # 2.  Groq API
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an intelligent document analysis assistant. "
                                "Answer the question based strictly on the provided Context only. "
                                "If the answer is not found in the context, say 'Information not found.' "
                                "Please provide the response in English."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Context: {context_text}\n\nQuestion: {user_input}"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                
                answer = completion.choices[0].message.content
                st.markdown(answer)

                # 3. Show References
                with st.expander("📍 References from PDF"):
                    for i, doc in enumerate(docs):
                        page = doc.metadata.get('page', 'Not found')
                        st.info(f"**Part {i+1} (Page {page}):**\n\n{doc.page_content[:400]}...")
                
                # Seve Answer
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {str(e)}")