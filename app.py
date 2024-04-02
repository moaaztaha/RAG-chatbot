import os
import shutil



from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

import chainlit as cl

from utils import load_process

TO_REMOVE = []

@cl.on_message
async def main(message: cl.Message):
    # if use uploads a file
   if len(message.elements) > 0:
      await cl.Message(content="Processing files...", disable_feedback=True).send()
      for file in message.elements:
        new_path = os.path.join("data", file.name)
        TO_REMOVE.append(new_path)
        shutil.copy(file.path, new_path)

      retriever = load_process()
      await cl.Message(content="Files processed successfully ✅", disable_feedback=True).send()
      await cl.Message(content=f"{retriever.get_relevant_documents(message.content)}", disable_feedback=True).send()





@cl.on_chat_end
def cleanup():
    # delete the data folder
    for path in TO_REMOVE:
        try:
            os.remove(path)
        except:
           print(f"Cloudn't delete file {path}!")