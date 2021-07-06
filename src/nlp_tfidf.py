from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
nltk.data.path.append('../nltk_data/')

corpus = ['hello, how are u?',
          'I\'m fine, how about u?',
          'I\'m very fine, my friend!']

tfidf_vect = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
tfidf_vect.fit(corpus)

corpus_transformed = tfidf_vect.transform(corpus)

print(f'Sparse matrix ( {type(corpus_transformed)} )\n')
print(corpus_transformed)
