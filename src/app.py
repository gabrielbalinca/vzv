import ollama
import os
import markdown

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 1. Load data from docx files
directory = os.path.abspath('train')

loader = DirectoryLoader(
    directory, 
    glob = "*.docx",
    loader_cls=UnstructuredWordDocumentLoader
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Prompt
def prompt(question, context):
    model='llama3'
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    messages=[{'role': 'user', 'content': formatted_prompt}]
    response = ollama.chat(
        model='llama3', 
        messages=[{
            'role': 'user', 
            'content': formatted_prompt}]
        )

    return response['message']['content']

# 3. setup (retrival augumented generation)
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = combine_docs(retrieved_docs)
    return prompt(question, formatted_context)

# 
with open('../tests/input.md', 'r') as f:
    markdown_text = f.read()

input = markdown.markdown(markdown_text)

with open('../tests/output.md', 'w') as f:
    f.write(rag_chain(input))