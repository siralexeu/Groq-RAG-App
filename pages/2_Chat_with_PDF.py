import streamlit as st

st.set_page_config(
    page_title="Chat with PDF 📄",
    page_icon="📄",
    layout="wide",
)

from groq import Groq
import os
import sqlite3
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils.chromadb_utils import create_or_get_collection, add_documents_to_collection, query_collection, \
    sanitize_collection_name
from utils.pdf_processing import extract_pdf_text, split_text_into_chunks

# Load environment variables
load_dotenv()

# Set chat type for this page
CHAT_TYPE = "pdf_chat"


# Initialize Groq client - delayed until after set_page_config
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets['GROQ_API_KEY'])


# Initialize embedding model (local, gratuit)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


def get_model_name():
    return st.secrets.get('MODEL_NAME', 'llama-3.3-70b-versatile')


def get_production_setting():
    return st.secrets.get('PRODUCTION', 'False')


def embed_text(text):
    """
    Generates an embedding using local Sentence Transformers (GRATUIT).
    """
    embedding_model = load_embedding_model()
    return embedding_model.encode(text).tolist()


def main():
    """
    Streamlit app for chatting with a PDF document.
    """
    # Get client and settings
    client = get_groq_client()
    MODEL_NAME = get_model_name()
    PRODUCTION = get_production_setting()

    # Ensure separate message storage for this page
    if f"{CHAT_TYPE}_messages" not in st.session_state:
        st.session_state[f"{CHAT_TYPE}_messages"] = []

    # Get messages specific to this page
    st.session_state.messages = st.session_state[f"{CHAT_TYPE}_messages"]

    # Check if in production mode
    is_production = PRODUCTION == "True"

    if is_production:
        if "api_key" not in st.session_state or not st.session_state.api_key:
            st.error("You do not have a valid API key. Please go back to the main page to enter one.")
            return

    st.title("Chat with PDF 📄")

    # Initialize session state
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = None
    if "previous_file" not in st.session_state:
        st.session_state.previous_file = None

    # Sidebar for clearing history
    with st.sidebar:
        st.header("Chat Settings")

        if st.button("Clear chat history"):
            st.session_state[f"{CHAT_TYPE}_messages"] = []
            st.success("Chat history has been cleared!")
            st.rerun()

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.previous_file:
            st.session_state.messages = []
            st.session_state.collection_name = None
            st.session_state.previous_file = uploaded_file.name

            with st.spinner("Processing the PDF..."):
                pdf_text = extract_pdf_text(uploaded_file)

                if st.checkbox("Show extracted text from PDF"):
                    st.text_area("Extracted Text", pdf_text, height=300)

                # Create or retrieve collection
                collection_name = sanitize_collection_name(f"pdf_{uploaded_file.name}")
                st.session_state.collection_name = collection_name
                collection = create_or_get_collection(collection_name)

                # Process PDF text and store embeddings
                if not collection.get()["documents"]:
                    chunks = split_text_into_chunks(pdf_text)
                    embeddings = [embed_text(chunk) for chunk in chunks]
                    add_documents_to_collection(collection, chunks, embeddings)
                    st.success("The PDF text has been indexed!")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Write a question about the PDF here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state[f"{CHAT_TYPE}_messages"] = st.session_state.messages

        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.collection_name:
            collection = create_or_get_collection(st.session_state.collection_name)
            query_embedding = embed_text(prompt)
            relevant_chunks = query_collection(collection, query_embedding)

            if relevant_chunks:
                context = "\n\n".join(relevant_chunks)
                prompt_with_context = f"Question: {prompt}\n\nRelevant content from PDF:\n{context}\n\nAnswer:"

                with st.chat_message("assistant"):
                    with st.spinner("Generating answer..."):
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system",
                                 "content": "You are an assistant that answers based on the PDF content."},
                                {"role": "user", "content": prompt_with_context},
                            ],
                            stream=True,
                        )

                        message_placeholder = st.empty()
                        full_response = ""

                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")

                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        st.session_state[f"{CHAT_TYPE}_messages"] = st.session_state.messages
            else:
                st.warning("No relevant content found in the PDF for this question.")


if __name__ == "__main__":
    main()