from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

CHROMA_DB_DIR = "./vector_db"
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    question: str

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = HuggingFaceHub(
    repo_id=MODEL_NAME,
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

@app.post("/ask")
async def ask_question(input: QueryInput):
    response = rag_chain({"query": input.question})
    return {
        "answer": response["result"],
        "sources": [doc.metadata.get("source", "") for doc in response["source_documents"]]
    }

@app.get("/")
def root():
    return {"message": "AgentFlow backend is running."}
