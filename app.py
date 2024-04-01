import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore  


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm( 
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF-GPT",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    gradient_text_html = """
    <style>
    .gradient-text{
        background: linear-gradient(74deg,#4285f4 0,#9b72cb 9%,#d96570 20%,#d96570 24%,#9b72cb 35%,#4285f4 44%,#9b72cb 50%,#d96570 56%,#131314 75%,#131314 100%);
        -webkit-background-clip: text;
        color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom:0px;
    }
    </style>

    <div class="gradient-text">
    Hello, Aman
    </div>
    """
    # Display the gradient text using markdown
    st.markdown(gradient_text_html, unsafe_allow_html=True)

    # st.header("Hello, Aman")
    st.header("How can I help you today with your PDFs?")
    # st.header("Chat with multiple PDFs :books:")

    card1 ='''
<style>
.card {
  /* Add shadows to create the "card" effect */
  box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
  transition: 0.3s;
    border-radius: 5px;
    height:200px;
    width:200px;
    background-color:#1e1f20;
}

/* On mouse-over, add a deeper shadow */
.card:hover {
  box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    background-color:#2f3133    ;

}

/* Add some padding inside the card container */
.container {
  padding: 2px 16px;
}
.cunt{
display:flex;
flex-direction:row;
    justify-content:space-between;
}
</style>
<div class="cunt">
    <div class="card">
  
  <div class="container">
    <span style='font-size:50px;'>&#128187;</span>
    <h4><b>Step 1</b></h4>    
    <p>Upload the PDFs in the sidebar</p>
  </div>
</div>


<div class="card">
  
  <div class="container">
  <span style='font-size:50px;'>&#128070;</span>
    <h4><b>Step 2</b></h4>
    <p>Click on process button and wait</p>
  </div>
</div>
<div class="card">
  
  <div class="container">
  <span style='font-size:50px;'>🔍</span>
    <h4><b>Step 3</b></h4>
    <p>Ask questions from the PDFs</p>
  </div>
</div>
</div>

'''
    st.markdown(card1, unsafe_allow_html=True)
    # st.write("H")
    st.markdown("<br>", unsafe_allow_html=True)

    # if pdf_docs:
    #  user_question = st.text_input("",placeholder="Enter a prompt here")
    #  if user_question:
    #   handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
    
    if pdf_docs:
     user_question = st.text_input("",placeholder="Enter a prompt here")
     if user_question:
      handle_userinput(user_question)
                
    





if __name__ == '__main__':
    main()
