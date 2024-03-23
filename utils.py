import os
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

def load_model_embeddings(MODEL: str, OPENAI_API_KEY: str):
    if MODEL.startswith("gpt"):
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=MODEL)
        embeddings = OpenAIEmbeddings()
        print(f"Using OpenAI model: {MODEL}")

        return model, embeddings
    
def build_prompt():
    template = """
    You are a helpful assistant call Mo. Answer the user question considering the chat history in less than 3 sentences.

    
    Chat History: {chat_history}

    User Question: {user_question}
    Answer:
    """

    prompt = PromptTemplate.from_template(template)
    return prompt

@st.cache_resource(show_spinner=False)
def create_chain(MODEL):
    with st.spinner("Loading model and creating embeddings..."):
        model, _ = load_model_embeddings(MODEL, os.environ["OPENAI_API_KEY"])
        prompt = build_prompt()

        chain = (
            {"user_question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
    return chain