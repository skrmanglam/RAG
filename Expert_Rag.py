# pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}
def get_pdf_text(input_data):
    text = ""
    for pdf in input_data:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        is_separator_regex= False,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(input_data):
    if isinstance(input_data, str):
        # get the text in document form
        loader = WebBaseLoader(input_data)
        document = loader.load()
        # split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)

    else: #pdf file
        document = get_pdf_text(input_data)
        # get the text chunks
        document_chunks_text = get_text_chunks(document)
        document_chunks = [Document(chunk) for chunk in document_chunks_text]

    embeddings = HuggingFaceInstructEmbeddings(
        query_instruction="Represent the query for retrieval: "
    )


    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    #vector_store = FAISS.from_documents(document_chunks, embeddings)

    return vector_store


def get_context_retriever_chain(vector_store):
    #llm = ChatOpenAI()
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
    )
    #llm_chain = LLMChain(prompt=prompt, llm=llm)

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    #llm = ChatOpenAI()
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']


# app config
st.set_page_config(page_title="Expert-RAG", page_icon="🤖")
st.title("Expert_RAG")

# sidebar
with st.sidebar:
    st.header("Input Options")

    # Input for Website URL
    website_url = st.text_input("Website URL")

    # Input for PDF Upload
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type = ['pdf'])
    #uploaded_pdf = st.file_uploader("Upload PDF")
    if st.button("Process"):
        with st.spinner("Processing"):
            input_data = pdf_docs

# Check if website URL or PDFs are provided
if website_url:
    input_data = website_url
elif pdf_docs:
    input_data = pdf_docs
else:
    st.info("Please enter a website URL or upload a PDF")
    input_data = None

# Process input data if available
if input_data:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore(input_data)

        # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)