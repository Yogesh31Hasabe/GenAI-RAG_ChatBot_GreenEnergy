from json import load
from logging import exception
import os
import openai
import sys
import shutil
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma


sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key  = os.environ['OPENAI_API_KEY']


class chat_gen():
    def __init__(self):
        self.chat_history=[]
        self.vectordb_path="dataset_docs/chroma"
        self.llm = self.get_llm()
        self.vectordb = self.doc_load_split_vector_store()


    def delete_directory_contents(self, directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Use shutil.rmtree to delete subdirectories recursively
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


    @st.cache_resource
    def get_llm(_self):
        llm_name = "gpt-3.5-turbo"
        print(f"Loading LLM: {llm_name}")
        llm = ChatOpenAI(model_name=llm_name, openai_api_key=openai.api_key, temperature=0)   
        print("LLM initialized")
        return llm
        

    def doc_load_split_vector_store(self):
        if(os.listdir(self.vectordb_path)!= []):
            self.delete_directory_contents(self.vectordb_path)
        loaders = []
        folder_path = "dataset_docs/green_energy_pdfs/"
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, filename)
                loader = PyPDFLoader(pdf_path)
                loaders.append(loader)
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
        print(f"Total no. of PDF files are : {len(loaders)}. \nTotal no. of pages of all PDF files are : {len(docs)}.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 24
        )
        splits = text_splitter.split_documents(docs)
        print(f"Total no. of Chunks created after splitting are : {len(splits)}.")

        embedding = OpenAIEmbeddings()
        persist_directory = 'dataset_docs/chroma/'
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
        print(f"Total no. of Collections stored in Chroma VectorDB are : {vectordb._collection.count()}.")
        return vectordb

    def load_model(self,):
        try:
            load_llm = self.llm
            load_vecdb = self.vectordb

            # Define your system instruction
            system_instruction = """ As an AI assistant, you must answer the query from the user from the retrieved content,
            if no relavant information is available, or if you don't know the answer, just say that you don't know, don't try to make up an answer."""

            # Define your template with the system instruction
            template = (
                f"{system_instruction} "
                "Combine the chat history{chat_history} and follow up question into "
                "a standalone question to answer from the {context}. "
                "Answer strictly in the given context. Don't try to make up an answer if you do not know."
                "Follow up question: {question}"
            )
            # Create a prompt template  
            prompt = PromptTemplate.from_template(template)

            chain = ConversationalRetrievalChain.from_llm(
                llm=load_llm,
                # retriever=self.doc_load_split_vector_store(folder_path).as_retriever(),
                retriever=load_vecdb.as_retriever(),
                combine_docs_chain_kwargs={'prompt': prompt},
                chain_type="stuff",
            )
            return chain
        except openai.RateLimitError as e:
            raise Exception(e)

    def ask_pdf(self,query):
        result = self.load_model()({"question":query,"chat_history": self.chat_history})
        self.chat_history.append((query, result["answer"]))
        #print(result)
        return result['answer']
    


if __name__ == "__main__":
    chat = chat_gen()