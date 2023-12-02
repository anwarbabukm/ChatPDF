# ChatPDF
The repository on application built for question answering on uploaded PDF using Retrieval Augmented Generation.

The application makes use of the quantized Llama 2 as the backend LLM. Quantization can dramatically reduce the memory and computation required to run the model by decreasing the precision of the model's parameters and activations. As a result, running the model on hardware with constrained resources could be achieved using quantized model.
It used the Langchain framework for the data processing and content retrieval from the vector store whereas streamlit is used for the frontend framework. 
The application lets you upload any type of PDFs after which the backend code extracts the content from it which is converted to embeddings. The embeddings are stored in Faiss vector store for the similarity matching based on the prompts. 
When a prompt is sent, it fetches the results from the vector store on similarity search and combines with the relevant document contents which is then send to the LLM for the text generation.

# Retrieval Augmented Generation Workflow

![RAG Workflow](https://github.com/anwarbabukm/ChatPDF/blob/main/RAG_Workflow.png)

# Application Screenshot
![Screenshot of the Application](https://github.com/anwarbabukm/ChatPDF/blob/main/Screenshot.jpg)
