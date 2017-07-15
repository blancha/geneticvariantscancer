import nltk
from nltk.corpus import stopwords

my_string = """My Character string."""

# NLTK's default English stopwords
default_stopwords = set(nltk.corpus.stopwords.words('english'))

# it can be enhanced
custom_stopwords = set(( u'(', u')'))

all_stopwords = default_stopwords | custom_stopwords

words = nltk.word_tokenize(my_string)

# Remove single-character tokens (mostly punctuation)
words = [word for word in words if len(word) > 1]

# Remove numbers
words = [word for word in words if not word.isnumeric()]

# Lowercase all words (default_stopwords are lowercase too)
words = [word.lower() for word in words]

# Remove stopwords
words = [word for word in words if word not in all_stopwords]

# Calculate frequency distribution
fdist = nltk.FreqDist(words)

# Output top 50 words

for word, frequency in fdist.most_common(300):
    print(u'{} : {}'.format(word, frequency))