import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()

# Set GROQ API Key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Set up prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to create embeddings and vector store
def create_vector_embedding():
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")
            st.session_state.docs = st.session_state.loader.load()

            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )

            # Filter out empty documents
            valid_docs = [doc for doc in st.session_state.final_documents if doc.page_content.strip()]
            if not valid_docs:
                st.error("All documents are empty after splitting. Please check your PDFs.")
                return

            # Create FAISS vector store
            st.session_state.vectors = FAISS.from_documents(valid_docs, st.session_state.embeddings)
            st.success("Vector Database is ready!")
        except Exception as e:
            st.error(f"Embedding creation failed: {str(e)}")
            raise

# Streamlit UI
st.title("RAG Document Q&A With Groq And Llama3")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()

# Handle Q&A
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please run 'Document Embedding' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"⏱️ Response time: {time.process_time() - start:.2f} seconds")

        st.markdown("### 🧠 Answer")
        st.write(response['answer'])

        with st.expander("📄 Document similarity search results"):
            for i, doc in enumerate(response.get('context', [])):
                st.markdown(f"**Chunk {i + 1}:**")
                st.write(doc.page_content)
                st.markdown('---')
