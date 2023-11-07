import streamlit as st
from streamlit_option_menu import option_menu
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
from src.utils import parse_xml
from src.utils import search_and_highlight
from src.utils import *
import re
from Bio import Entrez
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer

def zip_distribution(documents, top_of_word, keyword_search):
    file_path = f"save_img/zip_{keyword_search}_{top_of_word}.png"
    filtered_tokens = []
    if os.path.exists(file_path):
        st.markdown(f'<p style="text-align:center; color:red;">Table: Top {top_of_word} Words | Zipf Distribution of Terms</p>', unsafe_allow_html=True)
        st.image(file_path)
    else: 
      progress_text = "Please wait! Processing ..."
      my_bar = st.progress(0, text=progress_text)
      process = 0
      for doc in documents:
          tokens = clean_and_tokenize(doc)
          filtered_tokens.extend(tokens)
          process += int(100/len(documents))
          my_bar.progress( process  , text=progress_text)
      my_bar.empty()

      # Calculate word frequencies
      word_freq = Counter(filtered_tokens)

      # Sort the words by frequency in descending order
      sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

      # Display only the top 20 frequencies
      number_of_words = top_of_word
      top_words = dict(list(sorted_word_freq.items())[:number_of_words])

      # Set Seaborn style
      sns.set(style="whitegrid")

      # Plot the Zipf distribution for the top 20 words using Seaborn
      plt.figure(figsize=(10, 6))
      ax = sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), palette="viridis")
      ax.set(xlabel='Frequency', ylabel='Words')
      st.markdown(f'<p style="text-align:center; color:red;">Table: Top {number_of_words} Words | Zipf Distribution of Terms</p>', unsafe_allow_html=True)
      # plt.title(f'Top {number_of_words} Words | Zipf Distribution of Terms (with stopwords)')
      plt.tight_layout()

      # Display the plot
      st.pyplot(plt)
      plt.savefig(f"save_img/{keyword_search}_{top_of_word}.png", transparent=True)

def remove_stopwords(documents, top_of_word, keyword_search):
    # Remove stopwords and tokenize the text
    filtered_tokens = []
    file_path = f"save_img/stopwords_{keyword_search}_{top_of_word}.png"
    if os.path.exists(file_path):
        st.markdown(f'<p style="text-align:center; color:red;">Table: Top {top_of_word} Words | Zipf Distribution of Terms (Remove Stopwords)</p>', unsafe_allow_html=True)
        st.image(file_path)
    else: 
      progress_text = "Please wait! Processing ..."
      my_bar = st.progress(0, text=progress_text)
      process = 0
      for doc in documents:
          tokens = clean_and_tokenize(doc)
          filtered_tokens.extend([word for word in tokens if word not in stopwords.words('english')])
          process += int(100/len(documents))
          my_bar.progress( process  , text=progress_text)
      my_bar.empty()

      # Calculate word frequencies
      word_freq = Counter(filtered_tokens)
      # Sort the words by frequency in descending order
      sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
      # Display only the top 100 frequencies
      number_of_words = top_of_word
      top_100_words = dict(list(sorted_word_freq.items())[:number_of_words])

      # Set Seaborn style
      sns.set(style="whitegrid")
      # Plot the Zipf distribution for the top 100 words using Seaborn
      plt.figure(figsize=(10, 6))
      ax = sns.barplot(x=list(top_100_words.values()), y=list(top_100_words.keys()), palette="viridis")
      ax.set(xlabel='Frequency', ylabel='Words')
      # plt.title(f'Top {number_of_words} Zipf Distribution of Terms (Remove Stopwords)')
      st.markdown(f'<p style="text-align:center; color:red;">Table: Top {number_of_words} Words | Zipf Distribution of Terms (Remove Stopwords)</p>', unsafe_allow_html=True)
      plt.tight_layout()

      # Display the plot
      st.pyplot(plt)
      plt.savefig(file_path, transparent=True)

def porter_stemmer(documents, top_of_word, keyword_search, remove_stopwords = True): 
     
    # Apply Porter's stemming algorithm to the filtered tokens

    file_path = f"save_img/porter_{keyword_search}_{top_of_word}_{remove_stopwords}.png"
    if os.path.exists(file_path):
        st.markdown(f'<p style="text-align:center; color:red;">Table: Top {top_of_word} Words | Zipf Distribution of Terms (Stopwords Removed and Porter Stemming)</p>', unsafe_allow_html=True)
        st.image(file_path)
    else: 
        filtered_tokens = [] 
        if remove_stopwords:
            progress_text = "Please wait! Processing ..."
            my_bar = st.progress(0, text=progress_text)
            process = 0
            for doc in documents:
                tokens = clean_and_tokenize(doc)
                filtered_tokens.extend([word for word in tokens if word not in stopwords.words('english')])
                process += int(100/len(documents))
                my_bar.progress( process  , text=progress_text)
            my_bar.empty() 
        else:
            progress_text = "Please wait! Processing ..."
            my_bar = st.progress(0, text=progress_text)
            process = 0
            for doc in documents:
                tokens = clean_and_tokenize(doc)
                filtered_tokens.extend(tokens)
                process += int(100/len(documents))
                my_bar.progress( process  , text=progress_text)
            my_bar.empty() 
            
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        # Calculate word frequencies
        word_freq = Counter(stemmed_tokens)

        # Sort the words by frequency in descending order
        sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1], reverse=True))

        # Display only the top 20 frequencies
        number_of_words = top_of_word
        top_words = dict(list(sorted_word_freq.items())[:number_of_words])

        # Set Seaborn style
        sns.set(style="whitegrid")

        # Plot the Zipf distribution for the top 20 stemmed words using Seaborn
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=list(top_words.values()), y=list(top_words.keys()), palette="viridis")
        ax.set(xlabel='Frequency', ylabel='Words')
        #st.markdown(f'<p style="text-align:center; color:red;">Top {number_of_words} Zipf Distribution of Terms (with Stopwords Removed and Porter Stemming)</p>', unsafe_allow_html=True)
        #plt.title(f'Top {number_of_words} Zipf Distribution of Terms (with Stopwords Removed and Porter Stemming)')
        st.markdown(f'<p style="text-align:center; color:red;">Table: Top {number_of_words} Words | Zipf Distribution of Terms (Stopwords Removed and Porter Stemming)</p>', unsafe_allow_html=True)
        plt.tight_layout()
        plt.savefig(file_path, transparent=True)
        # Display the plot
        st.pyplot(plt)

def compare(documents,top_of_word, keyword_search, remove_stopwords = True):
    file_path = f"save_img/compare_{keyword_search}_{top_of_word}_{remove_stopwords}.png"
    if os.path.exists(file_path):
        st.markdown(f'<p style="text-align:center; color:red;">Table: Comparison Before and After Applying Porter Algorithm </p>', unsafe_allow_html=True)
        st.image(file_path)
    else: 
        
        filtered_tokens = [] 
        if remove_stopwords:
            progress_text = "Please wait! Processing ..."
            my_bar = st.progress(0, text=progress_text)
            process = 0
            for doc in documents:
                tokens = clean_and_tokenize(doc)
                filtered_tokens.extend([word for word in tokens if word not in stopwords.words('english')])
                process += int(100/len(documents))
                my_bar.progress( process  , text=progress_text)
            my_bar.empty() 
        else:
            progress_text = "Please wait! Processing ..."
            my_bar = st.progress(0, text=progress_text)
            process = 0
            for doc in documents:
                tokens = clean_and_tokenize(doc)
                filtered_tokens.extend(tokens)
                process += int(100/len(documents))
                my_bar.progress( process  , text=progress_text)
            my_bar.empty() 
            
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        # Calculate word frequencies before stemming
        word_freq_before = Counter(filtered_tokens)

        # Calculate word frequencies after stemming
        word_freq_after = Counter(stemmed_tokens)

        # Sort the words by frequency in descending order
        sorted_word_freq_before = dict(sorted(word_freq_before.items(), key=lambda item: item[1], reverse=True))
        sorted_word_freq_after = dict(sorted(word_freq_after.items(), key=lambda item: item[1], reverse=True))

        # Display only the top 20 frequencies
        number_of_words = top_of_word
        top_words_before = dict(list(sorted_word_freq_before.items())[:number_of_words])
        top_words_after = dict(list(sorted_word_freq_after.items())[:number_of_words])

        # Set Seaborn style
        sns.set(style="whitegrid")
        #st.markdown(f"<p style="text-align:center; color:red;">Table: Comparison Before and After Applying Porter's Algorithm </p>", unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center; color:red;">Table: Comparison Before and After Applying Porter Algorithm </p>', unsafe_allow_html=True)

        # Plot the Zipf distribution before and after Porter's stemming
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        ax = sns.barplot(x=list(top_words_before.values()), y=list(top_words_before.keys()), palette="viridis")
        ax.set(xlabel='Frequency', ylabel='Words')
        plt.title(f'Top {number_of_words} Zipf Distribution of Terms (Before Stemming)')

        plt.subplot(1, 2, 2)
        ax = sns.barplot(x=list(top_words_after.values()), y=list(top_words_after.keys()), palette="viridis")
        ax.set(xlabel='Frequency', ylabel='Words')
        plt.title(f'Top {number_of_words} Zipf Distribution of Terms (After Stemming)')

        plt.tight_layout()

        # Display the plots
        st.pyplot(plt)
        plt.savefig(file_path, transparent=True)
    
    
def frequency_analyst(): 
    st.image("image/analyst_title.png")
    keyword_search = ''
    keyword_search = st.text_input("Enter keyword :")
    edit_distance = st.toggle("Edit distance", value=False)
    path_keywords = os.listdir("dataset")
    if keyword_search in path_keywords: 
        st.info(f"Your keyword is {keyword_search}", icon="ℹ️")
    elif edit_distance and len(keyword_search) > 0: 
        suggestions = find_closest_keywords(keyword_search, path_keywords, num_suggestions = 10  )

        # Create a bar chart using Seaborn
        keywords, probabilities = zip(*suggestions)
        data = {"Keywords": keywords, "Probability": probabilities}
        df = pd.DataFrame(data)

        plt.figure(figsize=(8, 4))
        sns.set(style="whitegrid")  # Set the style to have a white grid
        ax = sns.barplot(x="Keywords", y="Probability", data=df)
        plt.xticks(rotation=45)
        plt.title("Keyword Probabilities with Edit Distance")

        # Add percentages on each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()*100:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')

        st.pyplot(plt)
        # st.write(f"Closest keywords to '{keyword_search}' is {keywords[0]}") 
        if probabilities[0] > 0.6: 
            st.info(f"Closest keywords to '{keyword_search}': {keywords[0]}",icon="ℹ️")
            keyword_search = keywords[0]
        else:
            st.warning("No found the keyword", icon="⚠️")
            keyword_search = '' 
    elif len(keyword_search) > 0: 
        st.warning("No found the keyword", icon="⚠️")
        keyword_search = ''
    if len(keyword_search) > 0: 
        st.sidebar.title("Setting")
        top_of_word = st.sidebar.number_input("Top of words", min_value=5, step=1, format="%d")
        documents = []
        list_file = os.listdir(f"dataset/{keyword_search}")
        for file in list_file: 
            documents.append(parse_xml_to_string(f"dataset/{keyword_search}/{file}")) 

        zip_distribution_status  = st.sidebar.toggle("Zipf Distribution", value=False) 
        if zip_distribution_status: 
            zip_distribution(documents, top_of_word, keyword_search)
            # Remove stopwords
            remove_stopwords_status  = st.sidebar.toggle("Remove Stopwords", value=False)
            if remove_stopwords_status:
                remove_stopwords(documents, top_of_word, keyword_search)

            porter_algorithm_status  = st.sidebar.toggle("Porter’s algorithm", value=False)
            if porter_algorithm_status:
                porter_stemmer(documents, top_of_word, keyword_search, remove_stopwords_status)

                compare_status  = st.sidebar.toggle("compare the difference", value=False)
                if compare_status: 
                    compare(documents, top_of_word, keyword_search, remove_stopwords_status)
                    st.balloons()