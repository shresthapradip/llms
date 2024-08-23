import sys

import sagemaker
from langchain_community.llms.sagemaker_endpoint import LLMContentHandler
from sagemaker.predictor import retrieve_default
from llama_index.embeddings.sagemaker_endpoint import SageMakerEmbedding
from langchain_community.embeddings import SagemakerEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.llms import SagemakerEndpoint
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains import create_extraction_chain
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv, find_dotenv
import requests
from collections import deque
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pathlib
import json
import openai
from typing import Dict
import os
import bs4
import string


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    parameters = {
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.3,
        "max_new_tokens": 256,
        "num_return_sequences": 4,
    }

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": self.parameters, **model_kwargs})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json['generated_text']


class RagCreator:
    def __init__(self, db_path: str, endpoint: str = None, llm_name: str = None):
        # if endpoint is None use open-ai
        if endpoint is None:
            print('Using openai llm')
            _ = load_dotenv(find_dotenv())  # read local .env file
            os.environ['USER_AGENT'] = \
                'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'

            openai.api_key = os.environ['OPENAI_API_KEY']
            if openai.api_key is None:
                raise Exception("Please set the OPENAI_API_KEY environment")
            self.embedding = OpenAIEmbeddings()
            self.llm = ChatOpenAI(model_name=llm_name, temperature=0)
            # Build prompt
            template = """You are an assistant for question-answering tasks specifically about the provided documents. "
                                "Use ONLY the following pieces of retrieved context to answer the question. "
                                "If you can't find the answer in the given context, say 'I'm sorry, but I couldn't find information about that in the provided context.' "
                                "Do not use any external knowledge. Use three sentences maximum and keep the answer concise. 
                                Always say "thanks for asking!" at the end of the answer. 
                                {context}
                                Question: {question}
                                Answer:"""
        else:
            print('Using sagemaker endpoint')
            sess = sagemaker.session.Session()
            self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            self.llm = SagemakerEndpoint(
                endpoint_name=endpoint,
                region_name=sess._region_name,
                model_kwargs={"temperature": 1e-10},
                content_handler=ContentHandler()
            )
            # Build prompt
            template = """Context information is below."
                        "{context}"
                        "Given the context information and not prior knowledge, answer the query."
                        "Query: {question}"
                        "Answer: """

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

        self.qa_chain_prompt = PromptTemplate.from_template(template)

        # create a vector store
        self.vectorstore = Chroma("langchain_store", self.embedding, persist_directory=db_path)

    def create_qa_chain(self, chain_type: str, k: int):
        # define retriever
        print(f"Creating {chain_type} for {self.get_num_docs()} docs")
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        # create a chatbot chain. Memory is managed externally.
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": self.qa_chain_prompt},
        )
        return qa_chain

    def get_num_docs(self):
        return self.vectorstore._collection.count()

    def load_web(self, url, limit=5):
        files_extension_to_exclude = ['.pdf', '.doc', '.docx', '.xls', '.xlsx',
                                      'jpg', '.jpeg', '.png', '.gif', '.ico',
                                      '.svg', 'webp']
        root = '/'
        nodes_list = deque()  # stack
        visited = set()  # set

        # add root
        nodes_list.append(root)

        # title and content only
        bs4_strainer = bs4.SoupStrainer(['h1', "article"])

        # set counter to 0
        counter = 0
        while len(nodes_list) > 0 and (counter < limit or limit == -1):
            counter = counter + 1
            node_str = nodes_list.pop()
            nodes = node_str.split('#')
            visited.add(nodes[0])
            full_node_url = url + nodes[0]
            print('Getting content from', full_node_url)
            loader = WebBaseLoader(
                web_paths=(full_node_url,),
                bs_kwargs={"parse_only": bs4_strainer},
            )
            loader.session.headers[
                'User-Agent'] = 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'

            # load and tokenize
            docs = loader.load_and_split(self.text_splitter)
            # to find all a tags
            scraped_doc = loader.scrape()
            # embed the docs
            self.vectorstore.add_documents(docs)
            # update nodes_list using anchor tags
            for a in scraped_doc.findAll('a'):
                # make sure the href exists and is internal
                if not a.has_attr('href') or not a['href'].startswith('/'):
                    continue
                # exclude files
                url_path = urlparse(a['href']).path
                file_extension = os.path.splitext(url_path)[1]
                if file_extension in files_extension_to_exclude:
                    continue
                # maintain stack
                anchor_strs = a['href'].split('#')
                if anchor_strs[0] not in visited and anchor_strs[0] not in nodes_list:
                    nodes_list.append(anchor_strs[0])

    def load_pdf(self, file):
        # load documents
        loader = PyPDFLoader(file)
        documents = loader.load()
        # split documents
        docs = self.text_splitter.split_documents(documents)
        # load documents
        self.vectorstore.add_documents(docs)
