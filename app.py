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
from src.search_engine import search_engine
from src.download_pubmed import download_pubmed
from src.frequency_analyst import frequency_analyst
from src.bag_of_word import * 
from src.skip_gram import * 

# Create an option menu for the main menu in the sidebar
st.set_page_config(page_title="Search PubMed Articles", page_icon="image/logo_csie2.png")
# st.image("image/title_search.png")
st.sidebar.image("image/logo_NCKU.jpeg", use_column_width=True)
with st.sidebar:
    selected = option_menu("Main Menu", ["Search Engine", "Download PubMed", "Frequency Analyst", "Continuous Bag of Word", "Skip-gram"],
                           icons=['search-heart-fill','cloud-arrow-down-fill', 'lightbulb-fill', "handbag-fill", "chat-left-text-fill"], menu_icon="bars", default_index=0)
# Based on the selected option, you can display different content in your web application
if selected == "Search Engine":
    search_engine()

elif selected == "Upload File":
    st.image("image/upload_file_title.png")

elif selected == "Download PubMed":
    download_pubmed()

elif selected == "Frequency Analyst":
    frequency_analyst()
elif selected == "Continuous Bag of Word":
    bag_of_word()
elif selected == "Skip-gram":
    skip_gram()
    


            








