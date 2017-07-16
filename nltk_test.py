#!/bin/python 

'''
Small script to test natural language processing using NLTK

Requires (3) argument: 
    - 1. [str] Path to the training dataset directory
    - 2. [int] Num of lines to parse
    - 3. [int] Max features to include
'''

# Import the required libraries
import re
import scipy as sp
import numpy as np
import nltk
import sys
from sklearn.feature_extraction.text import CountVectorizer

# Impoort relevant nltk tools
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Assign path to dataset
dir_path = sys.argv[1]

# Open file and read one line 
file_in = open(sys.argv[1] + '/training_text', 'r') 

# Prepare output list  variable
corpus = []

# Create stemmer object
ps = PorterStemmer()

# Create count vectorizer object
cv = CountVectorizer(max_features = int(sys.argv[3]))

# For loop to parse and clean file
for i in range(int(sys.argv[2])): 
    # Test to eliminate header
    line_in = file_in.readline().strip()
    # Elinimate header 
    if line_in[:2] == 'ID': 
        pass
    else: 
        # Eliminate all non-letter non-number characters
        clean_line = re.sub('[^a-zA-Z1-9]', ' ', line_in.split('||')[1])
        # Turn all characters to lowercase and split into words
        clean_line = clean_line.lower()
        clean_line = clean_line.split()
        # Remove all non-relevant words and stem words
        clean_line = [ps.stem(word) for word in clean_line if not word in set(stopwords.words('english')) and len(word) > 2] 
        clean_line = ' '.join(clean_line)
        corpus.append(clean_line)

X = cv.fit_transform(corpus).toarray()
