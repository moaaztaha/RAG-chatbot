from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def load_process(path: str = "data"):
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    print(f"# documents: {len(documents)} loaded!")

    # splittting the text
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len)

    texts = text_splitter.split_documents(documents)
    
    # embed and store persistently 
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embedding, persist_directory=persist_directory)
    vectorstore.persist()

    # make a retriever from the vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever