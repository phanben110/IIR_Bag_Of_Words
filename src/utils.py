import streamlit as st
import xml.etree.ElementTree as ET
import re
import os 
import editdistance
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import Counter


import xml.etree.ElementTree as ET

import xml.etree.ElementTree as ET

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []

    for article in root.findall('.//PubmedArticle'):
        # Initialize variables with empty strings
        pmid = ''
        title = ''
        abstract = ''
        journal_title = ''
        journal_issn = ''
        pubdate_year = ''
        pubdate_month = ''
        pubdate_day = ''
        author_list = []
        keyword_list = []

        pmid_element = article.find('.//PMID')
        if pmid_element is not None:
            pmid = pmid_element.text

        title_element = article.find('.//ArticleTitle')
        if title_element is not None:
            title = title_element.text

        abstract_element = article.find('.//Abstract/AbstractText')
        if abstract_element is not None:
            abstract = abstract_element.text

        # Extracting additional information
        journal_info = article.find('.//Journal')
        journal_title_element = journal_info.find('.//Title')
        if journal_title_element is not None:
            journal_title = journal_title_element.text

        journal_issn_element = journal_info.find('.//ISSN[@IssnType="Electronic"]')
        if journal_issn_element is not None:
            journal_issn = journal_issn_element.text

        pubdate = journal_info.find('.//PubDate')
        pubdate_year_element = pubdate.find('Year')
        pubdate_month_element = pubdate.find('Month')

        if pubdate_year_element is not None:
            pubdate_year = pubdate_year_element.text
        if pubdate_month_element is not None:
            pubdate_month = pubdate_month_element.text
        
        # Check if 'Day' element exists before accessing it
        pubdate_day_element = pubdate.find('Day')
        if pubdate_day_element is not None:
            pubdate_day = pubdate_day_element.text

        authors = article.find('.//AuthorList')
        if authors is not None:
            try:
                author_list = [f"{author.find('ForeName').text} {author.find('LastName').text}" for author in authors.findall('.//Author')]
            except AttributeError:
                author_list = []

        # Check if 'KeywordList' element exists before accessing it
        keyword_list_element = article.find('.//KeywordList[@Owner="NOTNLM"]')
        if keyword_list_element is not None:
            keyword_list = [keyword.text for keyword in keyword_list_element.findall('.//Keyword')]

        data.append({
            'PMID': pmid,
            'Title': title,
            'Journal Title': journal_title,
            'ISSN': journal_issn,
            'Publication Date': f"{pubdate_year}-{pubdate_month}-{pubdate_day}",
            'Abstract': abstract,
            'Authors': ', '.join(author_list),
            'Keywords': ', '.join(keyword_list)
        })

    return data



def search_and_highlight(article, search_term, case_sensitive=True):
    highlighted_fields = {}
    
    for key, value in article.items():
        flags = 0 if not case_sensitive else re.IGNORECASE
        try:
            highlighted_text = re.sub(
                fr'({re.escape(search_term)})',
                r'<span style="background-color: yellow">\1</span>',
                value,
                flags=flags,
            )
            if highlighted_text is not None:
                highlighted_fields[key] = highlighted_text
            else:
                highlighted_fields[key] = value
        except TypeError:
            # Handle the error by assigning the original value if 'value' is not a valid string
            highlighted_fields[key] = value

    return highlighted_fields

def parse_xml_to_string(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = []

    for article in root.findall('.//PubmedArticle'):
        # Initialize variables with empty strings
        pmid = ''
        title = ''
        abstract = ''
        journal_title = ''
        journal_issn = ''
        pubdate_year = ''
        pubdate_month = ''
        pubdate_day = ''
        author_list = []
        keyword_list = []

        pmid_element = article.find('.//PMID')
        if pmid_element is not None:
            pmid = pmid_element.text

        title_element = article.find('.//ArticleTitle')
        if title_element is not None:
            title = title_element.text

        abstract_element = article.find('.//Abstract/AbstractText')
        if abstract_element is not None:
            abstract = abstract_element.text

        # Extracting additional information
        journal_info = article.find('.//Journal')
        journal_title_element = journal_info.find('.//Title')
        if journal_title_element is not None:
            journal_title = journal_title_element.text

        journal_issn_element = journal_info.find('.//ISSN[@IssnType="Electronic"]')
        if journal_issn_element is not None:
            journal_issn = journal_issn_element.text

        pubdate = journal_info.find('.//PubDate')
        pubdate_year_element = pubdate.find('Year')
        pubdate_month_element = pubdate.find('Month')

        if pubdate_year_element is not None:
            pubdate_year = pubdate_year_element.text
        if pubdate_month_element is not None:
            pubdate_month = pubdate_month_element.text
        
        # Check if 'Day' element exists before accessing it
        pubdate_day_element = pubdate.find('Day')
        if pubdate_day_element is not None:
            pubdate_day = pubdate_day_element.text

        authors = article.find('.//AuthorList')
        if authors is not None:
            try:
                author_list = [f"{author.find('ForeName').text} {author.find('LastName').text}" for author in authors.findall('.//Author')]
            except AttributeError:
                author_list = []

        # Check if 'KeywordList' element exists before accessing it
        keyword_list_element = article.find('.//KeywordList[@Owner="NOTNLM"]')
        if keyword_list_element is not None:
            keyword_list = [keyword.text for keyword in keyword_list_element.findall('.//Keyword')]

        data.append({
            # 'PMID': pmid,
            'Title': title,
            # 'Journal Title': journal_title,
            # 'ISSN': journal_issn,
            # 'Publication Date': f"{pubdate_year}-{pubdate_month}-{pubdate_day}",
            'Abstract': abstract
            # 'Authors': ', '.join(author_list),
            # 'Keywords': ', '.join(keyword_list)
        })
    data_string = ""
    for entry in data:
        for key, value in entry.items():
            data_string += str(value)
    return data_string

# def find_closest_keywords(input_word, keyword_list, num_suggestions=3):
#     suggestions = []
#     for keyword in keyword_list:
#         distance = editdistance.eval(input_word, keyword)
#         suggestions.append((keyword, distance))
    
#     suggestions = sorted(suggestions, key=lambda x: x[1])[:num_suggestions]
#     return suggestions

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def find_closest_keywords_search_engine(input_word, keyword_list, num_suggestions=10):
    distances = [editdistance.eval(input_word, keyword) for keyword in keyword_list]
    softmax_probabilities = softmax(-np.array(distances))
    
    suggestions = list(zip(keyword_list, softmax_probabilities))
    suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)[:num_suggestions]
    
    return suggestions

def find_closest_keywords(input_word, keyword_list, num_suggestions=10):
    input_word = input_word.lower()
    keyword_list_nomal = keyword_list 
    keyword_list = [keyword.lower() for keyword in keyword_list]

    
    distances = [editdistance.eval(input_word, keyword) for keyword in keyword_list]
    softmax_probabilities = softmax(-np.array(distances))
    
    suggestions = list(zip(keyword_list_nomal, softmax_probabilities))
    suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)[:num_suggestions]
    
    return suggestions


def clean_and_tokenize(text):
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    return tokens