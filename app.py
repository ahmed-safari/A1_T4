import streamlit as st
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# Download necessary resources
nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Streamlit Title
st.title("Information Retrieval App with Word2Vec")

# Load Reuters Corpus
@st.cache_data
def load_and_preprocess_reuters():
    corpus_sentences = []
    documents = []
    for fileid in reuters.fileids():
        raw_text = reuters.raw(fileid)
        tokenized_sentence = [word for word in word_tokenize(raw_text) if word.isalnum()]
        corpus_sentences.append(tokenized_sentence)
        documents.append(raw_text[:200])  # Store first 200 chars for display
    return corpus_sentences, documents

corpus_sentences, documents = load_and_preprocess_reuters()
st.write(f"Loaded {len(documents)} documents.")

# Train Word2Vec Model
@st.cache_resource
def train_word2vec(corpus_sentences):
    return Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)

model = train_word2vec(corpus_sentences)

# Compute Document Embeddings
@st.cache_data
def compute_doc_embeddings(corpus_sentences, _model):
    def compute_avg_embedding(words):
        valid_words = [word for word in words if word in _model.wv]
        if not valid_words:
            return np.zeros(_model.vector_size)
        return np.mean([_model.wv[word] for word in valid_words], axis=0)
    
    return np.array([compute_avg_embedding(doc) for doc in corpus_sentences])

doc_embeddings = compute_doc_embeddings(corpus_sentences, model)
st.write("Document embeddings computed.")

# Query Processing
def process_query(query, _model):
    tokens = [word for word in word_tokenize(query.lower()) if word.isalnum()]
    return compute_avg_embedding(tokens, _model)

def compute_avg_embedding(words, _model):
    valid_words = [word for word in words if word in _model.wv]
    if not valid_words:
        return np.zeros(_model.vector_size)
    return np.mean([_model.wv[word] for word in valid_words], axis=0)

# Retrieve Top-k Documents
def retrieve_top_k(query_embedding, doc_embeddings, documents, k=5):
    if np.all(query_embedding == 0):
        return [("No relevant documents found.", 0.0)]

    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# Streamlit UI for Search
query = st.text_input("Enter your search query:")
if st.button("Search"):
    if query:
        query_embedding = process_query(query, model)
        results = retrieve_top_k(query_embedding, doc_embeddings, documents)
        st.write("### Top Relevant Documents:")
        for idx, (doc, score) in enumerate(results):
            st.write(f"**Document {idx+1}:** (Score: {score:.4f})")
            st.write(doc)
    else:
        st.warning("Please enter a query.")

# Word Embedding Visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key[:200]])
word_vectors_2d = tsne.fit_transform(word_vectors)

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])
st.pyplot(plt)
