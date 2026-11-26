import sys
import os
import time
from dotenv import load_dotenv

# Streamlit Imports
import streamlit as st
from streamlit.web import cli as stcli
from streamlit_extras.add_vertical_space import add_vertical_space

# LangChain Imports
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

# ------------------------------------------------------ PAGE CONFIG ------------------------------------------------------ #
load_dotenv()

st.set_page_config(page_title="News Chatbot", page_icon="📰", layout="centered")
st.header("News ChatBot 📰")

# ------------------------------------------------------ FUNCTIONS ------------------------------------------------------ #
def get_document_text_chunks(loader):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=80,
        length_function=len,
        separators=["\n\n", "\n", ".", " "]
    )
    return loader.load_and_split(text_splitter=text_splitter)

def get_vectorstore(docs):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        api_key=st.secrets["AZURE_OPENAI_API_KEY"],
        api_version="2023-05-15",
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"]
    )
    return FAISS.from_documents(documents=docs, embedding=embeddings)

def get_conversation_chain(vectorstore):
    # Use AzureChatOpenAI for gpt-3.5-turbo chat models
    llm = AzureChatOpenAI(
        azure_deployment="my-dna-gpt35turbo",
        api_key=st.secrets["AZURE_OPENAI_API_KEY"],
        api_version="2023-05-15",
        azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
        temperature=0.3
    )
    return RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

def initialize_session_state():
    defaults = {
        "gemini_pro_model": "gemini-pro",
        "messages": [],
        "conversation_chain": None,
        "urls": [],
        "is_processed": False,
        "URL_1": "",
        "URL_2": "",
        "URL_3": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session_state():
    # Clear URL text inputs safely
    st.session_state.URL_1 = ""
    st.session_state.URL_2 = ""
    st.session_state.URL_3 = ""
    
    # Reset app state variables
    st.session_state.messages = []
    st.session_state.conversation_chain = None
    st.session_state.urls = []
    st.session_state.is_processed = False

# ------------------------------------------------------ MAIN APP ------------------------------------------------------ #
def main():
    initialize_session_state()

    # ------------------------------------------------------ SIDEBAR ------------------------------------------------------ #
    with st.sidebar:
        st.subheader("News Article URLs")

        urls = []
        for i in range(3):
            url = st.text_input(
                f"URL {i+1}",
                key=f"URL_{i+1}",
                placeholder="Paste News URL here..."
            )
            if url.strip():
                urls.append(url.strip())

        process_button = st.button(label="Process URLs")

        if process_button:
            if not urls:
                st.warning("Please enter at least one valid URL.")
            else:
                try:
                    with st.spinner("Data Loading... ✅"):
                        loader = UnstructuredURLLoader(urls=urls)
                        docs = get_document_text_chunks(loader)

                    if not docs:
                        st.error("Could not extract any text from the provided URLs.")
                    else:
                        with st.spinner("Building Embedding Vector... ✅"):
                            vectorstore = get_vectorstore(docs)

                        with st.spinner("Building Conversation Chain... ✅"):
                            st.session_state.conversation_chain = get_conversation_chain(vectorstore)

                        # Persist processed state across reruns
                        st.session_state.urls = urls
                        st.session_state.is_processed = True
                        st.success("Done processing!", icon="✅")
                except Exception as e:
                    st.error(f"Error processing URLs: {e}")

        add_vertical_space(num_lines=1)

        st.markdown("""
        ### About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
        """)
        st.write("Made ❤️ by Lim Ming Jun")

        # on_click handles state reset; no st.rerun() needed
        st.button("Reset All", on_click=reset_session_state)

    # ------------------------------------------------------ MAIN LAYOUT ------------------------------------------------------ #
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your query:"):
        if not prompt.strip():
            return

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if st.session_state.is_processed and st.session_state.conversation_chain is not None:
                with st.spinner("Thinking..."):
                    result = st.session_state.conversation_chain.invoke({"question": prompt})
                
                answer = result.get("answer", "No answer generated.").strip()
                sources = result.get("sources", "").strip()
                assistant_response = f"{answer}\n\n**Sources:** {sources}" if sources else answer
            else:
                assistant_response = "Sorry, please input your news URL and click the **Process URLs** button before querying!"

            # Stream response while preserving newlines
            for word in assistant_response.split(" "):
                full_response += word + " "
                time.sleep(0.03)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response.strip())

        st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})

if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
