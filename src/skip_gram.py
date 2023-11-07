import streamlit as st
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import seaborn as sns

# Define a function to preprocess the text
def preprocess_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text_tokens = text.lower().split()
    text_tokens = list(set(text_tokens))
    return text_tokens

def train_skip_gram_model(data_file):
    # Đọc dữ liệu từ tệp và tạo một tệp nguồn dữ liệu LineSentence
    sentences = LineSentence(data_file)

    # Huấn luyện mô hình Skip Gram
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, sg=1)  # sg=1 để sử dụng Skip Gram

    # Lưu mô hình thành tệp word2vec_sg_model.bin
    model.save("word2vec_sg_model.bin")

def skip_gram():
    st.image("image/SG.png")
    
    # Training
    train_model = st.button("Train Model")
    if train_model:
        data_file = "tokenized_documents.txt"  # Thay đổi thành đường dẫn tệp của bạn
        train_skip_gram_model(data_file)
        st.session_state.trained = True  # Đánh dấu rằng mô hình đã được huấn luyện

    # Search
    keyword = st.text_input("Enter keyword to search for:")
    
    # Number of top similar words to display
    num_top_words = st.number_input("Number of Top Similar Words To Display", min_value=1, step=1, value=10)
    
    top_words = []  # Initialize top_words here

    search_word = st.button("Search")
    if search_word:

        # Load the Word2Vec model from the saved file
        with open("tokenized_documents.txt", "r") as file:
            text_data = file.read()
        keyword = keyword.lower()
        # Preprocess the text data and keyword
        text_tokens = preprocess_text(text_data)

        model = Word2Vec.load("word2vec_sg_model.bin")  # Load the Skip Gram model

        if keyword in model.wv:
            keyword_vector = model.wv[keyword]
        else:
            st.write(f"The keyword '{keyword}' is not in the vocabulary of the model.")
            return

        # Calculate cosine similarity with all words in text_data
        similarity_scores = []
        for word in text_tokens:
            if word in model.wv:
                word_vector = model.wv[word]
                similarity = cosine_similarity([keyword_vector], [word_vector])
                similarity_scores.append((word, similarity))

        # Sort the words by similarity score in descending order
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # Extract the specified number of top similar words and their similarity scores
        top_words = [word for word, similarity in similarity_scores[:num_top_words]]
        top_similarities = [similarity[0][0] for word, similarity in similarity_scores[:num_top_words]]

        st.markdown(f'Top {num_top_words} Word Similarities to "{keyword}":')
        for word, similarity in zip(top_words, similarity_scores[:num_top_words]):
            similarity_score = similarity[1][0].item()  # Convert to a scalar value
            st.write(f"{word} : {similarity_score:.2f}")

        # Create a Seaborn plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_similarities, y=top_words, palette='viridis')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Words')
        st.pyplot(plt)

        word_vectors = [model.wv[word] for word in top_words]  # Get vectors for top similar words
        pca = PCA(n_components=3)
        word_vectors_3d = pca.fit_transform(word_vectors)
        
        # Create a 3D scatter plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for word, (x, y, z) in zip(top_words, word_vectors_3d):
            if word == keyword:
                ax.scatter(x, y, z, c='red', marker='o', label=word)
                ax.text(x, y, z, word, fontsize=6, color='red', ha='right')
            else:
                ax.scatter(x, y, z, c='blue', marker='o', label=word)
                ax.text(x, y, z, word, fontsize=6, color='blue', ha='right')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(f'3D Visualization of Words related to "{keyword}"')
        # ax.legend()
        st.pyplot(fig)

