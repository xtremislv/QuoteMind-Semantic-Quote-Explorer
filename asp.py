import pandas as pd
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from arize.otel import register
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

# Load and preprocess data
df = pd.read_json("hf://datasets/Abirate/english_quotes/quotes.jsonl", lines=True)
df.dropna(inplace=True)
df['quote'] = df['quote'].str.lower()
df['author'] = df['author'].str.lower()
df['tags'] = df['tags'].apply(lambda x: [tag.lower() for tag in x])
df['combined'] = df.apply(lambda x: f"{x['quote']} [AUTHOR: {x['author']}] [TAGS: {', '.join(x['tags'])}]", axis=1)

# Load pre-trained model
model1 = SentenceTransformer(r"C:\Users\lovyv\Downloads\ai_ass\fine_tuned_quote_model")

# Encode data
embeddings = model1.encode(df['combined'].tolist(), show_progress_bar=True)
embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Quote retrieval function
def retrieve(query, top_k=5):
    query_emb = model1.encode([query])
    distances, indices = index.search(query_emb, top_k)
    results = []
    for i in indices[0]:
        results.append({
            "quote": df.iloc[i]['quote'],
            "author": df.iloc[i]['author'],
            "tags": df.iloc[i]['tags'],
            "similarity_score": float(distances[0][list(indices[0]).index(i)])
        })
    return results

# Gemini API setup
GoogleGenAIInstrumentor().instrument()

tracer_provider = register(
  project_name="your-next-llm-project",
  endpoint="http://localhost:6006/v1/traces",
  auto_instrument=True
)
genai.configure(api_key="AIzaSyC4G9dJu7fqD6iLrWtROSL_PWl0wGNEffc")
model = genai.GenerativeModel("gemini-1.5-flash")

# Generate structured answer
def generate_answer(query, context_quotes):
    context = "\n".join([
        f"Quote: {q['quote']} (Author: {q['author']}, Tags: {q['tags']})"
        for q in context_quotes
    ])
    prompt = f"""You are a semantic quote assistant. Use the following quotes to answer the query.

QUERY: {query}

CONTEXT:
{context}

STRUCTURED JSON OUTPUT:
"""
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("Semantic Quote Retriever")
query = st.text_input("Enter your query (e.g., quotes about courage by women authors):")

if query:
    retrieved = retrieve(query)
    st.json(retrieved)

    if st.button("Generate Structured Answer"):
        answer = generate_answer(query, retrieved)
        st.markdown("### Answer")
        st.markdown(answer)

