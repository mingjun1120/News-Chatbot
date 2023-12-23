# Streamlit Imports
import streamlit as st
from streamlit.web import cli as stcli
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Langchain Imports
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

# Import env variables
from dotenv import load_dotenv

# Import system
import sys
import os

# Import other modules
import pickle
import random
import time


# ------------------------------------------------------ FUNCTIONS ------------------------------------------------------ #
def get_document_text_chunks(loader):

    # Get the text chunks of the PDF file, accumulate to the text_chunks list variable becaus load_and_split() returns a list of Document
    docs = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len,
        separators= ["\n\n", "\n", ".", " "]
    ))

    return docs

def get_vectorstore(docs, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query", google_api_key=api_key)
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)

    return vectorstore

def get_conversation_chain(vectorstore):

    llm = GoogleGenerativeAI(model=st.session_state.gemini_pro_model, google_api_key=api_key, temperature=0.5)
    conversation_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, 
        retriever=vectorstore.as_retriever()
    )
    return conversation_chain

def initialize_session_state():
    # Set a default model
    if "gemini_pro_model" not in st.session_state:
        st.session_state["gemini_pro_model"] = "gemini-pro"
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "urls" not in st.session_state:
        st.session_state.urls = None
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = None
    if "is_vectorstore" not in st.session_state:
        st.session_state.is_vectorstore = False

def reset_session_state():
    # Use to clear the inputs in st.text_input for URL 1, URL 2, URL 3 when the user clicks the "Reset All" button
    st.session_state.URL_4, st.session_state.URL_5, st.session_state.URL_6 = st.session_state.URL_1, st.session_state.URL_2, st.session_state.URL_3
    st.session_state.URL_1, st.session_state.URL_2, st.session_state.URL_3 = '', '', ''
    
    # Delete all the keys in session state
    for key in st.session_state.keys():
        del st.session_state[key]
    
    # Initialize the default session state variables again
    initialize_session_state()

# ------------------------------------------------------ GLOBAL VARIABLES ------------------------------------------------------ #
# Load the environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Set the tab's title, icon and CSS style
page_icon = ":newspaper:"  # https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="News Chatbot", page_icon=page_icon, layout="centered")

# Page header
st.header(body=f"News ChatBot 📰")

def main():

    # Initialize the session state variables
    initialize_session_state()

    # ------------------------------------------------------ SIDEBAR ------------------------------------------------------ #
    # Sidebar contents
    with st.sidebar:

        st.subheader("News Article URLs")

        # Paste News URLs
        urls = []
        for i in range(3):
            url = st.text_input(f"URL {i+1}", key=f"URL_{i+1}", placeholder="Paste News URL here...")
            urls.append(url)
        urls = list(filter(None, urls)) # Remove None in the list

        # Process URLs
        process_button = st.button(label="Process URLs")

        if process_button:
            if urls != []:
                # st.session_state.clear() # Clear the session state variables
                with st.spinner(text="Data Loading...✅✅✅"):
                    # Load data
                    loader = UnstructuredURLLoader(urls=urls)
                
                with st.spinner(text="Text Splitting...✅✅✅"):
                    # Get the text chunks from the PDFs
                    docs = get_document_text_chunks(loader)

                with st.spinner(text="Building Embedding Vector...✅✅✅"):
                    # Create Vector Store
                    vectorstore = get_vectorstore(docs, api_key)
                    st.session_state.is_vectorstore = True

                with st.spinner(text="Building Conversation Chain...✅✅✅"):
                    # Create conversation chain
                    st.session_state.conversation_chain = get_conversation_chain(vectorstore)
                
                # Print System Message at the end
                st.success(body=f"Done processing!", icon="✅")

            # Use to check if URLs are processed. If not processed, users will be asked to upload PDFs when ask questions.
            st.session_state.is_processed = process_button

        if urls != []:
            st.session_state.urls = urls
        else:
            st.session_state.urls = None
            st.session_state.is_vectorstore = False

        add_vertical_space(num_lines=1)

        # Web App References
        st.markdown('''
        ### About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Gemini Pro](https://ai.google.dev/)
        ''')
        st.write("Made ❤️ by Lim Ming Jun")

        # Reset button part
        reset = st.button('Reset All', on_click=reset_session_state)
        if reset:
        #     for key in st.session_state.keys():
        #         del st.session_state[key]
        #     initialize_session_state()
            st.rerun()
    
    # ------------------------------------------------------ MAIN LAYOUT------------------------------------------------------ #
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Enter your query:"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            error_message = "Sorry, please input your news URL and click the Process button before querying!"

            if prompt != None and prompt.isspace() == False:
                if st.session_state.urls != None and st.session_state.is_processed != None and st.session_state.is_vectorstore == True:
                    # result will be a dictionary of this format --> {"answer": "", "sources": ""}
                    result = st.session_state.conversation_chain({"question": prompt}, return_only_outputs=True)
                    assistant_response = result.get("answer") + '\n\n**Sources:** ' + result.get('sources')

                    # Simulate stream of response with milliseconds delay
                    for chunk in assistant_response.split():
                        if chunk == "**Sources**:":
                            full_response = full_response + '\n\n' + chunk + " "
                        else:
                            full_response += chunk + " "
                        
                        time.sleep(0.05)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    assistant_response = error_message
                    # Simulate stream of response with milliseconds delay
                    for chunk in assistant_response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
            
                # # Add assistant response to chat history
                # st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                if prompt.isspace():
                    prompt = None
                
                if st.session_state.messages != [] and st.session_state.messages[-1]["content"] == error_message and prompt == None:
                    # Simulate stream of response with milliseconds delay
                    for chunk in error_message.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        # Add a blinking cursor to simulate typing
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Clear the user input after the user hits enter
            prompt = None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

