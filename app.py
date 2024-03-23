import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from utils import create_chain


# set page config
st.set_page_config(
    page_title="RAG Bot 🤖",
    layout="wide",
    page_icon="🤖"
)


# add title
st.title("RAG Bot 🤖")
st.write("Chat with your with your files using your favorite model!")
# streamlit collapsable widget for features
with st.expander("Features", expanded=True):
    st.write("1. **Choose between multiple models**")
    # st.write("1. **Chat with your files (pdfs)**")
    # st.write("2. **Histroy support**")

# init chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# add sidebar menu
with st.sidebar:
    st.title("Settings")

    OPENAI_MODELS = ["gpt-3.5-turbo", "gpt-4"]

    MODEL = st.selectbox("Model", OPENAI_MODELS)
    if MODEL in OPENAI_MODELS:
        st.markdown("### OpenAI API Key")
        OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
        if OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


    # clear history button
    if "chat_history" in st.session_state and st.button("Clear History"):
        st.session_state.chat_history = []

# show chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


if MODEL and OPENAI_API_KEY:
    # load chain
    chain = create_chain(MODEL)


    user_question = st.chat_input("Ask your question")
    if user_question is not None and user_question != "":
        with st.chat_message("Human"):
            st.markdown(user_question)
            st.session_state.chat_history.append(HumanMessage(content=user_question))
        
        with st.chat_message("AI"):
            response = st.write_stream(chain.stream({"user_question": user_question, "chat_history": st.session_state.chat_history}))
        st.session_state.chat_history.append(AIMessage(content=response))
