import os
import streamlit as st 
# Import the necessary libraries
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from src.utils import parse_xml
from src.utils import search_and_highlight
from src.utils import *
from nltk.stem import PorterStemmer
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# def make_context_vector(context, word_to_ix):
#     idxs = [word_to_ix[w] for w in context]
#     return torch.tensor(idxs, dtype=torch.long)

import torch
import torch.nn as nn

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMDEDDING_DIM = 100
model_name = "cbow_model_1.pth"


# raw_text = """We are about to study the idea of a computational process.
# Computational processes are abstract beings that inhabit computers.
# As they evolve, processes manipulate other abstract things called data.
# The evolution of a process is directed by a pattern of rules
# called a program. People create programs to direct processes. In effect,
# we conjure the spirits of the computer with our spells.""".split()

with open("tokenized_documents_test.txt", "r") as file:
    read_raw_text = file.read()

raw_text = read_raw_text.split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word:ix for ix, word in enumerate(vocab)}
ix_to_word = {ix:word for ix, word in enumerate(vocab)}

import json

# Load the word_to_id dictionary from the JSON file
with open('ix_to_word.json', 'r') as json_file:
    ix_to_word = json.load(json_file) 

with open('word_to_ix.json', 'r') as json_file:
    word_to_ix = json.load(json_file) 

# print(ix_to_word)

data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print("done 1")

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        #out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()
        
        #out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab_size)
        self.activation_function2 = nn.LogSoftmax(dim = -1)
        

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1,-1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.tensor([word_to_ix[word]])
        return self.embeddings(word).view(1,-1)



nltk.download('punkt') 
def bag_of_word():
    st.image("image/CBOW.png")
    st.markdown(f'<p style="text-align:center; color:red;">CBOW (Continuous Bag of Words) with window size = 2</p>', unsafe_allow_html=True)
    st.image("image/window_size_2.png")
    st.image("image/ex.jpg")

    st.markdown(f'<p style="text-align:center; color:red;">Model CBOW Architecture</p>', unsafe_allow_html=True)
    st.image("image/model_sum.png")
    st.markdown(f'<p style="text-align:center; color:red;">Training Loss</p>', unsafe_allow_html=True)
    st.image("image/loss.jpg")

    # Search
    keyword1 = st.text_input("Input context word W(t-2) :")
    keyword2 = st.text_input("Input context word W(t-1) :") 
    keyword3 = st.text_input("Input context word W(t+1) :")
    keyword4 = st.text_input("Input context word W(t+2) :")
    if len(keyword1) > 0 and len(keyword2) > 0 and len(keyword3) > 0 and len(keyword4) > 0:
        st.info(f"Your context of word W(t-2) = {keyword1}, W(t-1) = {keyword2}, W(t+1) = {keyword3}, W(t+2) = {keyword4}", icon="ℹ️")
    run_cbow = st.button("Run CBOW")
    if run_cbow:
        # Load the pre-trained CBOW model
        model = CBOW(vocab_size, EMDEDDING_DIM)
        model.load_state_dict(torch.load(model_name))  # Replace with the actual path to your saved model 
                # Create a Streamlit app
        # Display the model architecture
        # st.write('Model CBOW Architecture')
        
        # st.text(model)
        #TESTING
        context = [keyword1, keyword2, keyword3, keyword4]
        context_vector = make_context_vector(context, word_to_ix)
        a = model(context_vector)
        print(torch.argmax(a[0]).item())
        predict = ix_to_word[str(torch.argmax(a[0]).item())]

        context_vector = make_context_vector(context, word_to_ix)
        a = model(context_vector)
        import torch.nn.functional as F

        # Apply softmax to the a tensor
        softmax_scores = F.softmax(a[0], dim=0)

        # Get the top 10 words and their softmax probabilities
        top_k_values, top_k_indices = torch.topk(softmax_scores, k=10)

        # Convert indices to words using ix_to_word
        top_k_words = [ix_to_word[str(index.item())] for index in top_k_indices]

        # Convert top_k_values to a NumPy array
        top_k_values = top_k_values.detach().numpy()

        # Create a bar plot using Seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=top_k_values, y=top_k_words, palette="viridis")
        plt.xlabel("Probability")
        plt.ylabel("Word")
        plt.title("Top 10 Words Of Target")
        st.pyplot(plt)


        st.info(f"The target word: {predict}", icon="ℹ️")

        #st.write(f"Full sentence: {keyword1} {keyword2} {predict} {keyword3} {keyword4}", icon="ℹ️")
        st.write(f"Full sentence: <span style='background-color: yellow'>{keyword1} {keyword2} {predict} {keyword3} {keyword4}</span>", unsafe_allow_html=True, icon="ℹ️")


    st.sidebar.title("Visualization")
    visualization  = st.sidebar.toggle("Visualization 2D", value=False)

    if visualization: 
        # Load the pre-trained CBOW model
        model = CBOW(vocab_size, EMDEDDING_DIM)
        model.load_state_dict(torch.load(model_name))  # Replace with the actual path to your saved model

        # Extract word embeddings
        word_embeddings = model.embeddings.weight.data.numpy()

        # Reduce dimensionality to 2 using PCA
        pca = PCA(n_components=2)
        word_embeddings_2d = pca.fit_transform(word_embeddings)

        # Create a 2D scatter plot using Seaborn
        plt.figure(figsize=(12, 8))
        sns.set(style='whitegrid')

        # Convert word embeddings to a DataFrame
        import pandas as pd
        num_words_to_plot = st.sidebar.number_input("Number of word to plot", min_value=10, step=1, format="%d")
        # word_df = pd.DataFrame({
        #     'Word': list(ix_to_word.values()),
        #     'Dimension 1': word_embeddings_2d[:, 0],
        #     'Dimension 2': word_embeddings_2d[:, 1]
        # })

        # Create a DataFrame for the selected words
        word_df = pd.DataFrame({
            'Word': list(ix_to_word.values())[:num_words_to_plot],
            'Dimension 1': word_embeddings_2d[:num_words_to_plot, 0],
            'Dimension 2': word_embeddings_2d[:num_words_to_plot, 1]
        })

        # Create the scatter plot
        scatter = sns.scatterplot(x='Dimension 1', y='Dimension 2', data=word_df, hue='Word', palette='viridis', legend=False)

        # Annotate the points with word labels
        for line in range(0, word_df.shape[0]):
            scatter.text(word_df['Dimension 1'][line], word_df['Dimension 2'][line], word_df['Word'][line],
                        horizontalalignment='left', size='small', color='black')

        # Set labels for the axes
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        st.markdown(f'<p style="text-align:center; color:red;">2D Visualization of Word Embeddings</p>', unsafe_allow_html=True)
        st.pyplot(plt)



        