import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# Load variables from the .env file
load_dotenv()
api_key = os.getenv('API_KEY')

# Load the data
data = pd.read_csv("Mental_Health_FAQ.csv")

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for questions
st.write("Generating embeddings...")
embeddings = embedding_model.encode(data['Questions'].tolist())
embeddings = np.array(embeddings, dtype='float32')

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1,
    groq_api_key=api_key
)

# Streamlit App
st.title("Mental Health Assistant")
st.write("Ask your mental health-related questions below:")

# Textbox for user query
query = st.text_input("Enter your query:", "")

if query:
    # Generate embedding for the query
    query_embedding = np.array(embedding_model.encode([query]), dtype='float32')

    # Search for similar questions
    k = 3  # Number of nearest neighbors
    distances, indices = index.search(query_embedding, k)

    # Retrieve results and construct context
    results = [{"question": data['Questions'][i], "answer": data['Answers'][i]} for i in indices[0]]
    context_text = "\n".join([f"Q: {res['question']}\nA: {res['answer']}" for res in results])

    # Construct the prompt
    prompt = f"""
    You are a mental health assistant providing thoughtful and empathetic responses based on a context of similar questions and answers.

    Context:
    {context_text}

    User's Question:
    {query}

    Provide a helpful and supportive response based on the context.
    """

    # Generate the response
    with st.spinner("Generating response..."):
        response = llm.invoke(prompt)

    # Display the response
    st.subheader("Response:")
    st.write(response)

    # Optionally display the similar questions and answers
    st.subheader("Similar Questions and Answers:")
    for res in results:
        st.write(f"**Q:** {res['question']}\n**A:** {res['answer']}")
