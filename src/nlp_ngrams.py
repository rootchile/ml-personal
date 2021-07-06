from nltk.tokenize import word_tokenize
from nltk import ngrams
import nltk
nltk.data.path.append('../nltk_data/')

corpus = ['hello, how are u?',
          'I\'m fine, how about u?',
          'I\'m very fine, my friend!']

N = 3

tokenize = word_tokenize(corpus[0])
n_grams = list(ngrams(tokenize,N))
print(n_grams)