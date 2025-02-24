#!/usr/bin/env python
# coding: utf-8

# In[5]:


import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd


# In[6]:


nltk.download('reuters')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# In[8]:


# Streamlit Title
st.title("Information Retrieval App with Word2Vec")


# ## Step 1: Load the Reuters Corpus

# In[5]:


@st.cache_data
def load_and_preprocess_reuters():
    corpus_sentences = []
    documents = []
    for fileid in reuters.fileids():
        raw_text = reuters.raw(fileid)
        tokenized_sentence = [word for word in nltk.word_tokenize(raw_text) if word.isalnum() and word.lower()]
        corpus_sentences.append(tokenized_sentence)
        documents.append(raw_text[:200])  # Store first 200 chars for display
    return corpus_sentences, documents

corpus_sentences, documents = load_and_preprocess_reuters()
st.write(f"Loaded {len(documents)} documents from the Reuters corpus.")


# ## Step 2: Train a Word2Vec Model

# In[7]:


@st.cache_data
def train_word2vec(corpus_sentences):
    model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)
    return model

model = train_word2vec(corpus_sentences)


# # Step 3: Compute Document Embeddings

# In[ ]:


@st.cache_data
def compute_doc_embeddings(corpus_sentences, model):
    def compute_avg_embedding(words):
        valid_words = [word for word in words if word in model.wv]
        if not valid_words:
            return np.zeros(model.vector_size)
        return np.mean([model.wv[word] for word in valid_words], axis=0)
    
    return np.array([compute_avg_embedding(doc) for doc in corpus_sentences])

doc_embeddings = compute_doc_embeddings(corpus_sentences, model)
st.write("Document embeddings computed.")


# # Step 4: Process User Query

# In[ ]:


def process_query(query, model):
    tokens = [word for word in word_tokenize(query.lower()) if word.isalnum()]
    return compute_avg_embedding(tokens, model)

def compute_avg_embedding(words, model):
    valid_words = [word for word in words if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in valid_words], axis=0)


# # Step 5: Retrieve Top-k Documents

# In[ ]:


def retrieve_top_k(query_embedding, doc_embeddings, documents, k=5):
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]


# # Streamlit UI: Query Input and Search

# In[ ]:


query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query:
        query_embedding = process_query(query, model)
        results = retrieve_top_k(query_embedding, doc_embeddings, documents)

        # Display Results
        st.write("### Top Relevant Documents:")
        for idx, (doc, score) in enumerate(results):
            st.write(f"**Document {idx+1}:** (Score: {score:.4f})")
            st.write(doc)
    else:
        st.warning("Please enter a query to search.")


# ## Step 3: Extract Word Embeddings for Visualization

# In[9]:


import numpy as np
# Extract the learned word vectors and their corresponding words for visualization.
words = list(model.wv.index_to_key)[:200] # Limit to top 200 words for better visualization
word_vectors = np.array([model.wv[word] for word in words]) # Convert to NumPy array for compatibi


# ## Step 4: Reduce Dimensionality with t-SNE

# In[11]:


# Use t-SNE to project the high-dimensional word embeddings into a 2D space.
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
word_vectors_2d = tsne.fit_transform(word_vectors)


# ## Step 5: Visualize the Word Embeddings

# In[13]:


# Plot the 2D t-SNE visualization of the word embeddings with their labels.
def plot_embeddings(vectors, labels):
 plt.figure(figsize=(16, 12))
 for i, label in enumerate(labels):
        x, y = vectors[i]
        plt.scatter(x, y, color='blue')
        plt.text(x + 0.1, y + 0.1, label, fontsize=9)
 plt.title("Word2Vec Embeddings Visualized with t-SNE")
 plt.xlabel("t-SNE Dimension 1")
 plt.ylabel("t-SNE Dimension 2")
 plt.show()
plot_embeddings(word_vectors_2d, words)


# ## Key Questions for Students:
# 
# ### What do you observe about the clusters in the t-SNE plot?
# 
# - The clusters illustrate how words with similar meanings or usage patterns naturally group together, indicating that the model effectively captures semantic relationships.
# 
# 
# ### How do you think the choice of parameters (e.g., window size, vector size) affects the embeddings?
# 
# - Window size (5): A well-balanced choice for capturing contextual relationships in short to medium-length texts, such as news articles. It provides a sufficient contextual window without introducing excessive noise.
# 
# - Vector size (100): A practical trade-off between semantic richness and computational efficiency. It is adequate for medium-sized corpora like Reuters, but larger values could enhance performance for more complex linguistic tasks.
# 
# - Min count (5): Helps eliminate infrequent words, reducing noise and improving model focus on relevant terms. However, it might exclude valuable rare words, especially in domain-specific datasets.
# 
# - Workers (4): Efficient use of parallel processing to accelerate training, making it suitable for handling larger datasets.
# 
# ### What are the limitations of using Word2Vec and t-SNE for NLP tasks?
# 
# - Word2Vec: Faces difficulties in handling words with multiple meanings and requires substantial data for optimal performance.
# - t-SNE: Computationally expensive for large datasets and may introduce distortions in the visual representation of high-dimensional data.

# ## Information Retreival Task:
# ### Task: Build a Document Retrieval System using Word2Vec
# 1. Given a query string, and the most relevant documents from the Reuters corpus using
# Word2Vec embeddings.
# 2. Steps:
# - a. Preprocess the query string by tokenizing and removing stop words.
# - b. Compute the average Word2Vec embedding for the query string.
# - c. Compute the average Word2Vec embedding for each document in the Reuters corpus.
# - d. Use cosine similarity to and the top N most relevant documents for the query.
# 3. Display the top N document IDs and their similarity scores

# In[15]:


def process_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens 
                                


# In[17]:


corpus_sentences = []
doc_ids = []
for fileid in reuters.fileids():
    raw_text = reuters.raw(fileid)
    processed_text = process_text(raw_text)
    corpus_sentences.append(processed_text)
    doc_ids.append(fileid)
    
print(f"Number of documents in the Reuters corpus: {len(corpus_sentences)}")


# In[19]:


model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)
# Print vocabulary size
print(f"Vocabulary size: {len(model.wv.index_to_key)}")


# In[21]:


# b. Compute the average Word2Vec embedding for the query string.
def compute_avg_embedding(words, model):
    valid_words = [word for word in words if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[word] for word in valid_words], axis=0)


# In[23]:


doc_embeddings = np.array([compute_avg_embedding(doc, model) for doc in corpus_sentences])


# In[24]:


def retrieve_documents(query,top_n=5):
    query_tokens = process_text(query)
    query_embedding = compute_avg_embedding(query_tokens, model)
    
    #cosine similarity between query and alll document embedding
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    #the top N document indices sorted by similarity
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    # display the results
    print("\nTop relevant docs:")
    for idx in top_indices:
        doc_id = doc_ids[idx]
        doc_content = reuters.raw(doc_id)
        print(f"Document ID:{doc_id}, similarity score: {similarities[idx]:.4f}")
        print(f"Document content: {doc_content[:200]}")


# In[27]:


query = "stock market"
retrieve_documents(query, top_n=5)


# In[ ]:




