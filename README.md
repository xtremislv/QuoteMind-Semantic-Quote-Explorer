# QuoteMind: Semantic Quote Explorer

QuoteMind is a semantic quote retrieval application that lets you find meaningful quotes based on natural language queries. It combines cutting-edge technologies like sentence-transformer embeddings, FAISS vector search, and Google's Gemini model to deliver structured and contextual responses. The tool features a Streamlit-based user interface for easy interaction.

## 📺 Demo

[![Watch the Demo Video](https://img.youtube.com/vi/yukWh7swclg/0.jpg)](https://www.youtube.com/watch?v=yukWh7swclg)


## 🚀 Features

- 🔍 **Semantic Search**: Find quotes by meaning, not just keywords.
- 🧠 **Fine-Tuned Embeddings**: Uses a locally fine-tuned `SentenceTransformer` model.
- 📚 **Vector Indexing with FAISS**: Fast similarity-based search.
- 🗣️ **Structured Answers**: Uses Gemini to generate rich JSON responses.
- 🖥️ **Streamlit UI**: Clean and interactive web interface.

## 🛠️ Technologies Used

- **Python**
- **Pandas & NumPy**
- **FAISS**
- **Sentence Transformers**
- **Google Gemini (Generative AI)**
- **Streamlit**

## 🧑‍💻 How It Works

1. Loads and preprocesses quotes from a Hugging Face dataset.
2. Encodes each quote using a fine-tuned Sentence Transformer model.
3. Uses FAISS for indexing and fast semantic search.
4. Retrieves top quotes for a given query and displays them in Streamlit.
5. Optionally sends quotes to Gemini for structured output in JSON format.

## ▶️ Getting Started

### Prerequisites

- Python 3.8+
- API key for Google Gemini
- Streamlit

### Installation

```bash
git clone https://github.com/your-username/quote-mind.git
cd quote-mind
pip install -r requirements.txt
```
Run the App
```bash
streamlit run asp.py
```
