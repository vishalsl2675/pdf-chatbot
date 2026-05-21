import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# =====================================================
# LOAD ENV VARIABLES
# =====================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()


# =====================================================
# STREAMLIT PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="PDF Chatbot with ChromaDB",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stTextInput input {
    border-radius: 10px;
    border: 2px solid #FF4B4B;
    padding: 10px;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD LLM
# =====================================================

@st.cache_resource
def load_llm():

    model = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    return model


# =====================================================
# PDF READING
# =====================================================

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        try:

            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        except Exception as e:

            st.error(f"PDF Reading Error: {str(e)}")

    return text


# =====================================================
# TEXT CHUNKING
# =====================================================

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_text(text)

    return chunks


# =====================================================
# CREATE VECTOR STORE
# =====================================================

def create_vector_store(chunks):

    # DELETE OLD DATABASE
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    st.session_state.vector_store = vector_store


# =====================================================
# RETRIEVE RELEVANT CHUNKS
# =====================================================

def retrieve_relevant_chunks(question, top_k=8):

    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k}
    )

    docs = retriever.invoke(question)

    relevant_chunks = []

    for doc in docs:

        relevant_chunks.append(doc.page_content)

    return "\n\n".join(relevant_chunks)


# =====================================================
# USER QUESTION HANDLING
# =====================================================

def user_input(user_question):

    if "vector_store" not in st.session_state:

        st.warning("Please upload and process PDF first.")
        return

    try:

        relevant_text = retrieve_relevant_chunks(
            user_question,
            top_k=8
        )

        prompt = f"""
You are an advanced AI PDF assistant.

Your task is to answer ONLY using the provided PDF context.

GUIDELINES:
- Give detailed answers.
- Explain in simple language.
- Use headings and bullet points.
- Explain step-by-step if needed.
- Include examples from context.
- If answer is incomplete, clearly mention limitations.
- Do NOT hallucinate.
- If answer not found, say:
  "Answer is not available in the provided PDF."

CONTEXT:
{relevant_text}

QUESTION:
{user_question}

DETAILED ANSWER:
"""

        model = load_llm()

        response = model.invoke(prompt)

        st.subheader("📌 Answer")

        st.write(response.content)

    except Exception as e:

        st.error(f"Error: {str(e)}")


# =====================================================
# MAIN FUNCTION
# =====================================================

def main():

    st.title("Chat with PDF using Groq + ChromaDB 🚀")

    user_question = st.text_input(
        "Ask a Question from the PDF Files"
    )

    if user_question:

        with st.spinner("Generating Answer..."):

            user_input(user_question)

    # ================= SIDEBAR =================

    with st.sidebar:

        st.header("📂 Upload PDF Files")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):

            if pdf_docs:

                try:

                    with st.spinner("Processing PDFs..."):

                        # READ PDF
                        raw_text = get_pdf_text(pdf_docs)

                        if not raw_text.strip():

                            st.error("No text found in PDF")
                            return

                        # CREATE CHUNKS
                        text_chunks = get_text_chunks(raw_text)

                        # CREATE VECTOR STORE
                        create_vector_store(text_chunks)

                        st.success(
                            "PDF Processing Completed Successfully ✅"
                        )

                except Exception as e:

                    st.error(f"Processing Error: {str(e)}")

            else:

                st.warning("Please upload at least one PDF")


# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    main()