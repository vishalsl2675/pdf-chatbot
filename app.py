import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

from langchain_groq import ChatGroq


# ======================================================
# LOAD ENV VARIABLES
# ======================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Advanced PDF Chatbot",
    page_icon="📄",
    layout="wide"
)


# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.stTextInput input {
    border-radius: 10px;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# LOAD LLM
# ======================================================

@st.cache_resource
def load_llm():

    llm = ChatGroq(
        groq_api_key=groq_api_key,

        # BETTER MODEL FOR DETAILED ANSWERS
        model_name="llama-3.3-70b-versatile",

        temperature=0.3
    )

    return llm


# ======================================================
# LOAD EMBEDDINGS
# ======================================================

@st.cache_resource
def load_embeddings():

    embeddings = FastEmbedEmbeddings()

    return embeddings


# ======================================================
# READ PDF TEXT
# ======================================================

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


# ======================================================
# CREATE TEXT CHUNKS
# ======================================================

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(

        # LARGER CHUNKS
        chunk_size=1000,

        # BETTER CONTEXT OVERLAP
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ======================================================
# CREATE FAISS VECTOR STORE
# ======================================================

def create_vector_store(chunks):

    embeddings = load_embeddings()

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    st.session_state.vector_store = vector_store


# ======================================================
# RETRIEVE RELEVANT CHUNKS
# ======================================================

def retrieve_relevant_chunks(question):

    docs = st.session_state.vector_store.similarity_search(

        question,

        # MORE CHUNKS FOR BETTER ANSWERS
        k=6
    )

    relevant_chunks = []

    for doc in docs:

        relevant_chunks.append(
            doc.page_content
        )

    return "\n\n".join(relevant_chunks)


# ======================================================
# HANDLE USER QUESTION
# ======================================================

def user_input(user_question):

    if "vector_store" not in st.session_state:

        st.warning("Please upload and process PDF first.")
        return

    try:

        # RETRIEVE RELEVANT CONTEXT
        relevant_text = retrieve_relevant_chunks(
            user_question
        )

        # ADVANCED PROMPT
        prompt = f"""
You are an intelligent PDF assistant.

Your task is to answer the user's question ONLY using the provided context.

Instructions:
- Give detailed and well-structured answers.
- Explain concepts clearly in simple language.
- Use bullet points when needed.
- Include all important information from the context.
- Give examples if available in the context.
- Do not repeat sentences.
- If the answer is partially available, mention that clearly.
- Do not make up information outside the context.
- If the answer is not present in the context, say:
  "Answer is not available in the context."

Context:
{relevant_text}

Question:
{user_question}

Detailed Answer:
"""

        # LOAD MODEL
        model = load_llm()

        # GENERATE RESPONSE
        response = model.invoke(prompt)

        # DISPLAY ANSWER
        st.subheader("Detailed Reply")

        st.write(response.content)

    except Exception as e:

        st.error(f"Error: {str(e)}")


# ======================================================
# MAIN FUNCTION
# ======================================================

def main():

    st.title("Chat with PDF using Groq + FAISS 🚀")

    st.write(
        "Upload PDF files and ask detailed questions from the documents."
    )

    # USER QUESTION
    user_question = st.text_input(
        "Ask a Question from the PDF Files"
    )

    if user_question:

        with st.spinner("Generating Detailed Answer..."):

            user_input(user_question)

    # ==================================================
    # SIDEBAR
    # ==================================================

    with st.sidebar:

        st.header("Menu")

        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):

            if pdf_docs:

                try:

                    with st.spinner("Processing PDFs..."):

                        # EXTRACT TEXT
                        raw_text = get_pdf_text(pdf_docs)

                        if not raw_text.strip():

                            st.error("No text found in PDF")
                            return

                        # CREATE CHUNKS
                        text_chunks = get_text_chunks(
                            raw_text
                        )

                        # CREATE VECTOR STORE
                        create_vector_store(
                            text_chunks
                        )

                        st.success(
                            "PDF Processing Completed Successfully"
                        )

                except Exception as e:

                    st.error(
                        f"Processing Error: {str(e)}"
                    )

            else:

                st.warning(
                    "Please upload at least one PDF"
                )


# ======================================================
# RUN APP
# ======================================================

if __name__ == "__main__":

    main()