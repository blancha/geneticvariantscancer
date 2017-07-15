import numpy as np
import pandas as pd

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

DATA_PATH = "./dataset/"
train_variants = pd.read_csv(DATA_PATH + "training_variants")
train_text = pd.read_csv(DATA_PATH + "training_text", delimiter="\|\|")

# construct training set
X = train_text["ID,Text"]
y = train_variants["Class"]

# Make features
vectorizer = CountVectorizer(min_df=1, stop_words=stopwords.words('english'))

# train_X contains the bag of words vectorized form in a numpy array, all ready for PCA
train_X = vectorizer.fit_transform(X).toarray()
