from gc import callbacks
import os
import shutil
import time
import chainlit as cl
from utils import load_process
from chainlit.input_widget import Select
from langchain_community.vectorstores import Chroma
TO_REMOVE = []


@cl.on_chat_start
async def init():
    await cl.Message(content="Initializing...", disable_feedback=True).send()
    chain = load_process()
    cl.user_session.set("chain", chain)
    await cl.Message(content="Your assistant is ready 😊", disable_feedback=True).send()
    
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Choose the prefered model",
                values=["gpt-3.5-turbo", "gemma:2b", "llama2", "gpt-4"],
                initial_index=0,
            )
        ]
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # if use uploads a file
    if len(message.elements) > 0:
        await cl.Message(content="Processing files...",
                         disable_feedback=True).send()
        for file in message.elements:
            if file.type != "application/pdf":
                await cl.Message(content=f"File {file.name} is not a pdf. Skipping...",
                         disable_feedback=True).send()
                continue
            new_path = os.path.join("data", file.name)
            TO_REMOVE.append(new_path)
            shutil.copy(file.path, new_path)

        chain = load_process()
        cl.user_session.set("chain", chain)
        await cl.Message(content="Files processed successfully ✅",
                         disable_feedback=True).send()


    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    response = await chain.acall(message.content, callbacks=[cb])
    result = response["result"]
    source_documents = response["source_documents"]

    text_elements = []
    if source_documents:
        for source_doc in source_documents:
            source_name = source_doc.metadata["source"].split("/")[-1]
            text_elements.append(cl.Text(content=source_doc.page_content, name=source_name))
            
        source_names = [text_el.name for text_el in text_elements]

    if source_names:
            result += f"\nSources: {', '.join(source_names)}"
    else:
        result += "\n\nNo sources found"

    await cl.Message(content=result, elements=text_elements).send()


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    
    Chroma().delete_collection()
    chain = load_process(model=settings["Model"])
    cl.user_session.set("chain", chain)
    await cl.Message(content="Swtiched to model: " + settings["Model"] + " ✅",
                         disable_feedback=True).send()


@cl.on_chat_end
def cleanup():
    # delete db directory
    shutil.rmtree("database", ignore_errors=True)

    # delete the data folder
    for path in TO_REMOVE:
        try:
            os.remove(path)
        except:
            print(f"Cloudn't delete file {path}!")