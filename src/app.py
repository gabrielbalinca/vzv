import ollama
import docx
import os
import streamlit

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 1. Load data from docx files
directory = os.fsencode("./train")
def getText():
    for encodedFile in os.listdir(directory):
        file = os.fsdecode(encodedFile)
        if file.endswith(".docx"): 
            doc = docx.Document(file)
            fullText = []
            for para in doc.paragraphs:
                fullText.append(para.text)
            continue
        else:
            continue
    return '\n'.join(fullText)

loader = TextLoader(getText())
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Load model
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 3. Prompt
def prompt(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', 
                           messages=[{
                               'role': 'user', 
                               'content': formatted_prompt}])
    return response['message']['content']

# 4. RAG
retriever = vectorstore.as_retriever()
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return prompt(question, formatted_context)

# 5. Streamlit
streamlit.title('app.py')
result=streamlit.text_input("Ask your question")
if result:
    streamlit.write(prompt(result))