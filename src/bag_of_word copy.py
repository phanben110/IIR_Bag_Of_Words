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

nltk.download('punkt') 
def bag_of_word():
    st.image("image/CBOW.png")
    # Search
    keyword = st.text_input("Input context word:")
    case_sensitive = True
    # case_sensitive = st.toggle("Case Sensitive Search", value=True)
    matching_articles = []
    # edit_distance = st.toggle("Edit distance", value=False)
    edit_distance = True
    load_file = False

    if edit_distance and len(keyword) > 0: 
        data = []
        documents = []
        if load_file == False: 
            # Load uploaded files
            filtered_tokens = [] 
            list_file = os.listdir("dataset/brain injury")
            for file in list_file:
                try:
                  data += parse_xml(f"dataset/brain injury/{file}")
                  documents.append(parse_xml_to_string(f"dataset/brain injury/{file}"))
                except: 
                  continue 

            tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
            # Save tokenized_documents to a text file
            with open("tokenized_documents.txt", "w") as file:
                for document in tokenized_documents:
                    file.write(" ".join(document) + "\n")

            for doc in documents:
              tokens = clean_and_tokenize(doc)
              filtered_tokens.extend([word for word in tokens if word not in stopwords.words('english')])

            len(filtered_tokens)
            unique_list = list(set(map(str.lower, filtered_tokens)))

            filtered_list = [word for word in unique_list if not word.isnumeric() and word.isalpha() and len(word) <= 10]

            with open("filtered_list.txt", "w") as file:
                for word in filtered_list:
                    file.write(word + "\n")
        else:
           # Read filtered_list from a text file
            filtered_list = []
            with open("filtered_list.txt", "r") as file:
                for line in file:
                    word = line.strip()  # Remove newline characters
                    filtered_list.append(word)

            # Read tokenized_documents from a text file
            tokenized_documents = []
            with open("tokenized_documents.txt", "r") as file:
                for line in file:
                    tokens = line.strip().split()  # Split the line into tokens
                    tokenized_documents.append(tokens)



        # Calculate similarity scores for all keywords
        similarity_scores = [(word, fuzz.ratio(keyword, word)) for word in filtered_list]
        # Sort the list of keywords by similarity in descending order
        sorted_keywords = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        # Get the top 10 keywords
        top_10_keywords = [word for word, score in sorted_keywords[:9]]
        top_10_keywords.append(keyword)

        option = st.selectbox('Recommend the nearest word...',top_10_keywords)

        st.info(f"Your keyword: {option}", icon="ℹ️")
        keyword = option    

        
        # Create a Word2Vec model using CBOW
        if load_file == False: 
            model = Word2Vec(sentences=tokenized_documents, window=5, min_count=1, sg=0)
            model.save("word2vec_model_cbow.model")
        else: 
            model = Word2Vec.load("word2vec_model_cbow.model")
        similar_words =model.wv.most_similar(keyword)
        st.write("Output target word: ")
        st.write(similar_words)
        # To find similar words to a given word:


