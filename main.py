import streamlit as st
import time

# LangChain Modules
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



# Retrieve API Key from Streamlit Secrets
GOOGLE_API_KEY = st.secrets["api_key"]

st.title("📖 RAG Application built on Gemini Model")

# Load and process PDF
loader = PyPDFLoader("yolov9_paper.pdf")
data = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

# Create vector database
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Define LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, google_api_key=GOOGLE_API_KEY)

# Define system prompt as a string
system_prompt: str  = (
    "You are an AI assistant specializing in answering questions based on retrieved documents. "
    "Use the provided context to generate a response. If the answer isn't available, say you don't know. "
    "Limit responses to three concise sentences.\n\n{context}"
)

# Ensure correct type for ChatPromptTemplate
chat_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Use chat_prompt in the document chain
question_answer_chain = create_stuff_documents_chain(llm, chat_prompt)

# Create Retrieval Chain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Maintain chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
query = st.chat_input("💬 Ask something about the document...")

# Process query if input is given
if query:
    response = rag_chain.invoke({
        "question": query,
        "chat_history": st.session_state.chat_history
    })
    
    # Display response
    st.write("🤖 **AI Response:**")
    st.write(response["answer"])

    # Update chat history
    st.session_state.chat_history.append((query, response["answer"]))
