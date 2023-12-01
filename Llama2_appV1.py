
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings 

DB_FAISS_PATH = 'vectorstore/db_faiss'
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def extract_text_with_langchain_pdf(pdf_file):
    loader = UnstructuredFileLoader(pdf_file)
    documents = loader.load()
    #print(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return texts

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 2000,
        temperature = 0.2
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response['result']

# Assume you have functions like extract_text_with_langchain_pdf and final_result defined elsewhere

def main():
    st.title("PDF Text Extractor Chatbot :male-technologist:")

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file 👇", type=["pdf"])

    last_uploaded_filename = st.session_state.get('last_uploaded_filename', None)
    run_embedding = False  # Flag to control the embedding process

    if uploaded_file is not None:
        st.write("The uploaded file is: ",uploaded_file.name)
        file_details = {
            "Filename": uploaded_file.name
        }

        # Check if the uploaded PDF is different from the last uploaded PDF
        if uploaded_file.name != last_uploaded_filename:
            with open(uploaded_file.name, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            run_embedding = True  # Set the flag to True when the document is different

            # Update the last uploaded filename in the session state
            st.session_state.last_uploaded_filename = uploaded_file.name

    if run_embedding:
        with st.spinner('Embeddings are in process...'):
            extract_text_with_langchain_pdf(uploaded_file.name)
            st.success('Embeddings are created successfully!')

    st.markdown("<h4 style='color:black;'>Chat Here</h4>", unsafe_allow_html=True)
    chat_history = st.session_state.get('chat_history', [])
    user_input = st.text_input("You:", key="input")

    # Search the database for a response based on user input and update session state
    if user_input:
        query = user_input
        answer = final_result(query)

        # Append the current interaction to the chat history
        chat_history.append({'user': query, 'bot': answer})

        # Update the chat history in the session state
        st.session_state.chat_history = chat_history

        # Display the current answer with a grey background
        st.subheader("Current Answer:")
        st.markdown('<div style="background-color: #f0f0f0; padding: 10px; margin-bottom: 10px;">{}</div>'.format(answer), unsafe_allow_html=True)

    # Display the chat history
    st.markdown("<h4 style='color:black;'>Chat History</h4>", unsafe_allow_html=True)
    for interaction in chat_history[:-1]:
        st.write(f"**You:** {interaction['user']}")
        st.write(f"**Bot:** {interaction['bot']}")
        st.write("---")

if __name__ == "__main__":
    main()