from langchain.document_loaders import DirectoryLoader
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Initialize LLM with phi-3.5-mini
llm = HuggingFaceHub(repo_id="huggingface/phi-3.5-mini", model_kwargs={"temperature": 0})

# Load text files from a local directory
loader = DirectoryLoader("./notes", glob="*.txt")
documents = loader.load()

# Create embeddings and vector store from documents
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# Setup a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Listen to command line input and print answers
while True:
    query = input("Enter your query (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa_chain.run(query)
    print("Answer:", answer)
