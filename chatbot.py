# import os
# from pinecone import Pinecone
# from llama_index.embeddings.gemini import GeminiEmbedding
# from llama_index.core import StorageContext, VectorStoreIndex, download_loader
# from llama_index.llms.gemini import Gemini
# from llama_index.vector_stores.pinecone import PineconeVectorStore
# from llama_index.core import Settings
# from dotenv import load_dotenv

# load_dotenv()
 
# # Configure Gemini embedding model and settings
# embed_model = GeminiEmbedding(model_name="models/embedding-001") 

# llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])

# pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# pinecone_index = pinecone_client.Index("constitution")
# Settings.llm = llm
# Settings.embed_model = embed_model
# Settings.chunk_size = 1024

# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# documents = SimpleDirectoryReader("law-gpt\data").load_data()

# vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# #Create a storage context using the created Vector Store
# storage_context = StorageContext.from_defaults(
#     vector_store=vector_store
# )
# #Use the chunks of documents and the storage context to create the index
# index = VectorStoreIndex.from_documents(
#     documents,
#     storage_context=storage_context
# )

# chat_engine = index.as_chat_engine()
# while True:
#     text_input = input("User: ")
#     if text_input.lower() == "exit":
#         break
#     try:
#         response = chat_engine.chat(text_input)

#         print(f"Agent: {response.response}")

#     except Exception as e:
#         print(f"Error: {str(e)}")

import os
import streamlit as st
from pinecone import Pinecone
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit App Interface
st.title("Indian Constitution Q&A")

st.write("""
    Ask any question about the Indian Constitution, and I'll try to provide the answer based on the document.
    Type 'exit' to end the conversation.
""")

# Function to initialize Pinecone, LLM, and the document index
def initialize_app():
    # Configure Gemini embedding model and settings
    embed_model = GeminiEmbedding(model_name="models/embedding-001")
    llm = Gemini(api_key=os.environ["GOOGLE_API_KEY"])
    pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pinecone_client.Index("constitution")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024

    # Load documents and set up the index
    documents = SimpleDirectoryReader("law-gpt\data").load_data()
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create an index for the documents
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Create a chat engine from the index
    chat_engine = index.as_chat_engine()

    return chat_engine

# Initialize the app only once
if 'chat_engine' not in st.session_state:
    st.session_state.chat_engine = initialize_app()

# Query input box
text_input = st.text_input("Your Question:")

if text_input:
    if text_input.lower() == "exit":
        st.write("Exiting the chat. Goodbye!")
    else:
        try:
            # Get the response from the chat engine
            response = st.session_state.chat_engine.chat(text_input)
            st.write(f"Agent: {response.response}")
        except Exception as e:
            st.write(f"Error: {str(e)}")
