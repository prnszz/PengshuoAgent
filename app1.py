import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import chromadb
import os
import glob
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# 设置页面配置
st.set_page_config(page_title="Chat with Pengshuo", layout="wide")

# 自定义CSS样式
import streamlit as st

# 首先添加主题检测
st.markdown("""
<style>
    /* 消息容器基础样式 */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        width: 100%;
        overflow: hidden;
    }
            
    /* 消息内容样式 */
    .message-content {
        width: 100%;
        text-align: right !important;
    }
            
    /* 用户消息样式 - 深色主题 */
    [data-theme="dark"] .user-message {
        background-color: #2b313e;
        color: white;
        margin-left: auto;
        margin-right: 0;
        border-radius: 15px 15px 0 15px;
        display: inline-block;
        max-width: 80%;
        text-align: right;
        float: right;
        clear: both;
    }
    
    /* 用户消息样式 - 浅色主题 */
    [data-theme="light"] .user-message {
        background-color: #e6e9ef;
        color: #0f1116;
        margin-left: auto;
        margin-right: 0;
        border-radius: 15px 15px 0 15px;
        display: inline-block;
        max-width: 80%;
        text-align: right;
        float: right;
        clear: both;
        border: 1px solid #d1d5db;
    }
    
    /* 助手消息样式 - 深色主题 */
    [data-theme="dark"] .assistant-message {
        background-color: #1e1e1e;
        color: white;
        margin-right: 20%;
        border-radius: 15px 15px 15px 0;
    }

    /* 助手消息样式 - 浅色主题 */
    [data-theme="light"] .assistant-message {
        background-color: #f3f4f6;
        color: #0f1116;
        margin-right: 20%;
        border-radius: 15px 15px 15px 0;
        border: 1px solid #d1d5db;
    }

    /* 清除按钮样式 - 深色主题 */
    [data-theme="dark"] .stButton>button {
        background-color: #4e4e4e;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        margin: 1rem 0;
    }

    /* 清除按钮样式 - 浅色主题 */
    [data-theme="light"] .stButton>button {
        background-color: #e2e8f0;
        color: #1a202c;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: 1px solid #cbd5e0;
        margin: 1rem 0;
    }

    /* 输入框样式 - 深色主题 */
    [data-theme="dark"] .stTextInput>div>div>input {
        background-color: #2b313e;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: 1px solid #4e4e4e;
    }

    /* 输入框样式 - 浅色主题 */
    [data-theme="light"] .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #1a202c;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

st.title("Try to ask something about Pengshuo 😃")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Add this line to initialize show_welcome
if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True


def clear_chat_history():
    """Clear the chat history from session state"""
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

def load_pdfs(pdf_files):
    """Load multiple PDF files and return combined documents"""
    all_docs = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = os.path.basename(pdf_file)
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Sorry, may be try again later?")
    return all_docs

@st.cache_resource
def initialize_rag():
    try:
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory="./chroma_db",
            )
        )
        pdf_files = glob.glob("*.pdf")
        all_docs = load_pdfs(pdf_files)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        splits = text_splitter.split_documents(all_docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        collection_name = "pdf_collection"
        
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        if len(vectorstore._collection.get()['ids']) == 0:
            vectorstore.add_documents(splits)
        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None
        )
        
        return retriever, llm, pdf_files
    
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        raise e

# Initialize the system
try:
    retriever, llm, pdf_files = initialize_rag()
    
    # Add clear chat history button with custom styling
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("Clear Chat History"):
            clear_chat_history()

    # Show welcome message if it's the first visit and no chat history
    if st.session_state.show_welcome and len(st.session_state.chat_history) == 0:
        welcome_container = st.container()
        with welcome_container:
            st.info("""
            👋 Welcome! Here are some questions you can ask about Pengshuo:
            
            • When will you graduate?\n
            • Tell me about one paper you published\n
            • What's your major?\n
            • What is your research focus?\n
            • What are your language skills?\n
            
            Feel free to ask any questions about Pengshuo's background, research, or experience!
            """)
    
    # Create a container for chat messages
    chat_container = st.container()
    
    # Display chat history with custom styling
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-content">You: <br>{message['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div>Pengshuo's agent: <br>{message['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Create a container for the input
    input_container = st.container()
    
    with input_container:
        query = st.chat_input("Ask a question about Pengshuo or his paper (English plz🥺)")
    
    if query:
        # Hide welcome message when user starts chatting
        st.session_state.show_welcome = False
        try:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            system_prompt = """
                You are an AI agent representing Pengshuo Qiu. Here's your key information:

                IDENTITY & BACKGROUND:
                - Full name: Pengshuo Qiu (仇鹏硕 or キュウ)
                - Current status: 3rd year student at Tohoku University, will be graduating in March 2026
                - Advisor: Yuichiroh Matsubayashi
                - Affiliation: Tohoku EduNLP Lab and Tohoku NLP Group
                - Email: qiu.pengshuo.t5@dc.tohoku.ac.jp
                - Research focus: Natural Language Processing (NLP) with emphasis on Large Language Models (LLM)

                PUBLICATIONS (3 papers):
                1. "MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?"
                Authors: Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, Hongsheng Li

                2. "Scenarios and Approaches for Situated Natural Language Explanations"
                Authors: Pengshuo Qiu, Frank Rudzicz, Zining Zhu

                3. "MMSearch: Benchmarking the Potential of Large Models as Multi-modal Search Engines"
                Authors: Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanmin Wu, Jiayi Lei, Pengshuo Qiu, Pan Lu, Zehui Chen, Guanglu Song, Peng Gao, Yu Liu, Chunyuan Li, Hongsheng Li

                EXPERIENCE:
                - Research Intern at ECAL, SIT (Contributing to Situated Natural Language Explanations paper)
                - Research Intern at MMLAB, CUHK (Contributing to MathVerse and MMSearch papers)
                - Administrative Assistant at Tohoku Global Learning Center (Oct 2022 - Jan 2023)
                - Exchange programs: University of York (Feb-Mar 2023), UC Davis (Aug-Sep 2022)

                SKILLS:
                - Languages: English, Chinese, Japanese
                - Programming: Python, Copilot, PyTorch, TensorFlow, Hugging Face Transformers

                Use the above information and the following retrieved context to answer questions. Always provide complete, consistent answers, especially when discussing publications or experience. If you don't know something or if it's not mentioned in the provided context, say that you don't know.

                {context}
                """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            with st.spinner("Generating response..."):
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                
                # Add assistant's response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            # Rerun to update the chat display
            st.rerun()
        
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            
except Exception as e:
    st.error("Failed to initialize the system. Please check your configuration and try again.")