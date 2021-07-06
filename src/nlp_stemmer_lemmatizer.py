from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

import nltk
nltk.data.path.append('../nltk_data/')

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

words = ['buying','buy','bought']

for word in words:
    print(f'word: {word}, stemmed: {stemmer.stem(word)}, lemma = {lemmatizer.lemmatize(word)}')