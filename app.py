import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        file_path = "medical_data.csv"

        if not os.path.exists(file_path):
            st.error(f"❌ File '{file_path}' not found.")
            return []

        loader = CSVLoader(
            file_path=file_path,
            source_column="output",
            encoding="utf-8"
        )

        docs = loader.load()

        if not docs:
            st.warning("No documents found in CSV.")
            return []

        splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        return splitter.split_documents(docs)

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []


# Create vector database
@st.cache_resource
def create_vectorstore(_documents):
    if not _documents:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(_documents, embeddings)


# Load Groq LLM
@st.cache_resource
def load_llm():
    try:
        api_key = st.secrets["GROQ_API_KEY"]

        return ChatGroq(
            model="llama3-70b-8192",
            api_key=api_key,
            temperature=0.2,
            max_retries=2,
        )

    except Exception:
        st.error("🚫 GROQ_API_KEY not found in Streamlit Secrets.")
        return None


# Setup RAG chain
@st.cache_resource
def setup_rag_chain(_vectorstore, _llm):
    if _vectorstore is None or _llm is None:
        return None

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an expert medical assistant.

Use ONLY the provided medical text to answer the question.

Question:
{question}

Medical Text:
{context}

Provide the following information in simple English:

Symptoms:
Treatment:
Impact on Pregnancy:
Mental Health Impact:
Age Group Affected:

Answer:
"""
    )

    retriever = _vectorstore.as_retriever()

    return RetrievalQA.from_chain_type(
        llm=_llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )


# Streamlit UI
st.set_page_config(
    page_title="Medical Data ChatBot",
    layout="centered"
)

st.title("🩺 Medical Data ChatBot")

documents = load_data()
vectorstore = create_vectorstore(documents)
llm = load_llm()
rag_chain = setup_rag_chain(vectorstore, llm)

query = st.text_input(
    "Enter a disease name to extract information:"
)

if query:
    if rag_chain is None:
        st.error(
            "⚠️ System not ready. Check CSV file, dependencies, and API key."
        )
    else:
        with st.spinner("🔍 Searching..."):
            try:
                response = rag_chain.run(query)

                st.subheader("📋 Extracted Information")
                st.write(response)

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Powered by Groq Llama 3, FAISS, LangChain and Sentence Transformers")
