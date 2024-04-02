import os
from httpx import stream
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA





def load_process(path: str = "data", model: str = "gpt-3.5-turbo"):
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    print(f"# documents: {len(documents)} loaded!")

    # splittting the text
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len)

    texts = text_splitter.split_documents(documents)
    
    # choose llm and embedding
    if model.startswith("gpt"):
        embedding = OpenAIEmbeddings()
        llm=OpenAI(temperature=0, streaming=True, model_name=model)

    else:
        llm = Ollama(model=model, temperature=0)
        embedding = OllamaEmbeddings(model=model)

    print("************ Switched to model:", model, "************")
    
    # embed and store persistently 
    # persist_directory = 'database'
    vectorstore = Chroma.from_documents(texts, embedding)
    # vectorstore.persist()

    # make a retriever from the vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    

    # chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # chain_type_kwargs=chain_type_kwargs,
    )


    return chain