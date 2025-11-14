import streamlit as st

st.set_page_config(
    page_title="Simple Chat with LLM",
    page_icon="💬",
    layout="wide",
)

from groq import Groq
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Set chat type for this page
CHAT_TYPE = "simple_chat"

# Initialize Groq client as cached resource
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets['GROQ_API_KEY'])

def get_model_name():
    return st.secrets.get('MODEL_NAME', 'llama-3.3-70b-versatile')

def get_production_setting():
    return st.secrets.get('PRODUCTION', 'False')


def generate_response(prompt, client, model_name):
    """
    Streams a response using Groq's chat model.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an assistant that answers users' questions in English."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        full_response = ""
        response_placeholder = st.empty()

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        return f"An error occurred: {str(e)}"


def main():
    """
    Streamlit app for a simple chat interface with LLM.
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

    # Check if we are in production mode
    is_production = PRODUCTION == "True"

    if is_production:
        if "api_key" not in st.session_state or not st.session_state.api_key:
            st.error("You do not have a valid API key. Please return to the main page to enter one.")
            return

    # Initialize session state for history
    if "history" not in st.session_state:
        st.session_state.history = []
    if "selected_session_index" not in st.session_state:
        st.session_state.selected_session_index = None

    # If no previous chat history exists, create a new session
    if not st.session_state.history:
        new_session = {
            "messages": [],
            "name": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state.history.append(new_session)
        st.session_state.selected_session_index = 0

    # Sidebar for session history
    with st.sidebar:
        st.header("Session History")

        session_options = [
            f"{i + 1}. {session['name']}"
            for i, session in enumerate(st.session_state.history)
        ]

        selected_session = st.selectbox(
            "Select a session",
            options=session_options,
            index=st.session_state.selected_session_index or 0,
            key="session_selectbox",
        )

        new_session_index = int(selected_session.split(".")[0]) - 1
        if new_session_index != st.session_state.selected_session_index:
            st.session_state.selected_session_index = new_session_index
            st.session_state.messages = st.session_state.history[new_session_index]["messages"]
            st.session_state[f"{CHAT_TYPE}_messages"] = st.session_state.messages
            st.rerun()

        # Start a new session
        if st.button("Start a new session"):
            st.session_state.messages = []
            new_session = {
                "messages": [],
                "name": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.history.append(new_session)
            st.session_state.selected_session_index = len(st.session_state.history) - 1
            st.session_state[f"{CHAT_TYPE}_messages"] = []
            st.rerun()

        # Clear session history
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.messages = []
            st.session_state[f"{CHAT_TYPE}_messages"] = []
            st.session_state.selected_session_index = None
            st.success("Chat history has been cleared!")
            st.rerun()

    st.title("Simple Chat with LLM 💬")

    # Display chat messages
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Store user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state[f"{CHAT_TYPE}_messages"] = st.session_state.messages

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response in streaming mode
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = generate_response(prompt, client, MODEL_NAME)

        # Store the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state[f"{CHAT_TYPE}_messages"] = st.session_state.messages

        # Save messages to the correct session
        if st.session_state.selected_session_index is not None:
            st.session_state.history[st.session_state.selected_session_index][
                "messages"] = st.session_state.messages.copy()


if __name__ == "__main__":
    main()