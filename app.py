import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Deepseek-R1-Distill-Llama-70b")

# Prompt template for QA
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

# Function to create embeddings and vector DB
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Load PDFs from folder
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit app UI
st.title("📄 RAG Document Q&A With Groq And Deepseek-R1")

user_prompt = st.text_input("🔍 Enter your query about the research paper")

if st.button("📚 Generate Document Embedding"):
    with st.spinner("Processing documents and generating embeddings..."):
        create_vector_embedding()
    st.success("✅ Vector Database is ready!")

# Run retrieval and QA only if vectors exist
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please generate the document embeddings first by clicking the 'Generate Document Embedding' button.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed = time.process_time() - start
        print(f"Response time: {elapsed:.2f} seconds")

        st.write("### 💬 Answer")
        st.write(response['answer'])

        with st.expander("📎 Document Similarity Matches"):
            for i, doc in enumerate(response['context']):
                st.markdown(f"**Match {i+1}:**")
                st.write(doc.page_content)
                st.write("---")
