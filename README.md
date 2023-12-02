# ChatPDF
The repository on application built for question answering on uploaded PDF using the quantized Llama 2.

The application makes use of the quantized Llama 2 as the backend LLM. It used the Langchain framework for the data processing and streamlit for the frontend framework.
It lets you upload any pdf into the application and extracts the content and stores it in the vector store. Based on the questions being asked, it fetches the results from the vector store on similarity search and combines with the prompt to be sent to LLM for the text generation.
# Retrieval Augmented Generation Workflow

![RAG Workflow](https://github.com/anwarbabukm/ChatPDF/blob/main/RAG_Workflow.png)

# Application Screenshot
حشحAA
![Screenshot of the Application](https://github.com/anwarbabukm/ChatPDF/blob/main/Screenshot.jpg)
