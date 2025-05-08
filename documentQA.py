from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Load a local file
loader = TextLoader("Pantri.txt", encoding="utf-8")
docs = loader.load()

# Create embeddings with Ollama
embeddings = OllamaEmbeddings(model='llama3')

# Store and index in Chroma
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()

# Connect the retriever and model
llm = Ollama(model='llama3')
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a question
response = qa_chain.run("What is this document about?")
print(response)
