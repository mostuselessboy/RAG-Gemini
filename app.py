# PDF READING
import os
from PyPDF2 import PdfReader

# STREAMLIT
import streamlit as st

# GOOGLE
import google.generativeai as genai

# LANGCHAIN
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS # WORKS ON FAISS INSTEAD OF CHROMADB , WILL USE GPU
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# ENVIRONMENT
from dotenv import load_dotenv # Uncomment these Lines to add using Environment else you can add manually
# load_dotenv()
# os.getenv("GOOGLE_API_KEY")


#<<<< VARIABLES TO CHANGE >>>>>
GOOGLE_GEMINI_API_KEY = "AIzaSyAuo39Tdn6eWUYBcpXhM3LRTn67ycVqbx0"
pdf_docs = ["main.pdf",]


genai.configure(api_key=GOOGLE_GEMINI_API_KEY)



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text



def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks 



def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question from the context in points by rephrasing it , dont provide the answer as it is. GIVE ANSWER ONLY IN POINTS LEAVING A LINE BETWEEN EACH POINTS!, if the answer is not in
    provided context just say, "Please be more concise with your question🙏", don't provide the wrong answer. \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.6, 
                                    safety_settings={
                                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE
                                    }
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    try:
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True, )
    except Exception:
        return {'output_text':["AI Cannot Answer these type of Questions for Safety Reason"]}
    print(response)
    return response


def main():
    st.set_page_config(
        page_title="Delhi Police Bot",
        page_icon="🤖"
    )


    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .viewerBadge_link__qRIco{display:none;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    with st.spinner("Processing..."):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

    st.image("botheader.png", caption="", use_column_width=True)

    st.title("👮Delhi Police ChatBot💬")
    st.write("दिल्ली पुलिस आपकी सेवा में 🙏")
    st.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Delhi Police Seva mein aapka swagat hai 🙏"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                st.write(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()

