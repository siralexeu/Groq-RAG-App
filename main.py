import streamlit as st

PRODUCTION = st.secrets.get("PRODUCTION", "False")

st.set_page_config(
    page_title="Streamlit RAG App",
    page_icon="💬",
    layout="wide"
)

def main():
    st.title("Welcome to the application 💬")
    st.write("""
        This application has two main functionalities:
        1. **Simple chat**: interact with the LLM in a simple, conversational way
        2. **Chat with PDF files**: upload a PDF file and interact with the LLM to extract information from it
    """)

    if PRODUCTION == "True":
        st.info("Production mode is enabled, but Groq runs locally and does not require an API key.")
    else:
        st.info("Development mode: All pages are available without API authentication.")

if __name__ == "__main__":
    main()
