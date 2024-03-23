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
    Answer the following question in a maximum of 3 sentences.

    Question: {question}
    """

    prompt = PromptTemplate.from_template(template)
    return prompt

@st.cache_resource(show_spinner=False)
def create_chain(MODEL):
    with st.spinner("Loading model and creating embeddings..."):
        model, _ = load_model_embeddings(MODEL, os.environ["OPENAI_API_KEY"])
        prompt = build_prompt()

        chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
    return chain