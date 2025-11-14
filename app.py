from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from src.helper import download_embeding
from langchain_pinecone import PineconeVectorStore
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
from src.prompt import system_prompt
import os
from pinecone import Pinecone
from flask import Flask, render_template, jsonify, request

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

app=Flask(__name__)

index_name='medicalchatbot'
embedding=download_embeding()

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

retiver=docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain=create_stuff_documents_chain(model,prompt)
rag_chain=create_retrieval_chain(retiver,question_answer_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a question.", 400

    response = rag_chain.invoke({"input": msg})
    answer = response.get("answer", "Sorry, I couldn't generate a response.")
    print("Response:", answer)
    return answer




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)

