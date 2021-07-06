import io #stream handling
import numpy as np
import pandas as pd
import time
# data nltk path
import nltk
nltk.data.path.append('../nltk_data/')

from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

def load_vectors(fname):
    f_in = io.open(fname,'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, f_in.readline().split())
    data = {}
    for line in f_in:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def sentence_to_vec(s, embeddings_dict, stop_words, tokenizer):
    """
    This function returns embedding for the whole sentence.
    
    :param s: sentence (string)
    :param embeddings_dict: dictionary word:vector
    :param stop_words: list of stop_words, if any
    :param tokenizer: a tokenization function
    """
    
    words = str(s).lower()
    words = tokenizer(words)
    # remove stop words, if any, and only alpha-numeric tokens
    words = [w for w in words if not w in stop_words and w.isalpha()]
    
    embeddings = []
    for w in words:
        if w in embeddings_dict:
            embeddings.append(embeddings_dict[w])
    
    # dimensions = 300
    if len(embeddings)==0:
        return np.zeros(300)

    # list of embeddings to array
    embeddings = np.array(embeddings)

    # normalized vector
    sum = embeddings.sum(axis=0)
    return sum/np.sqrt((sum**2).sum())


if __name__=='__main__':
    
    time_init = time.time()
    print('Loading training data')
    df = pd.read_csv('../data/imdb/imdb.csv')
    print(f'Loaded: {round(time.time()-time_init)} secs.')
    
    # target
    df.sentiment = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    df['kfold'] = -1
    # randomize
    df = df.sample(frac=1).reset_index(drop=True)
    
    # load embeddings
    time_init = time.time()
    print('Loading embeddings')
    embeddings = load_vectors('../embeddings_data/fasttext/crawl-300d-2M.vec')
    print(f'Loaded: {round(time.time()-time_init)} secs.')
    
    time_init = time.time()
    print('Creating sentence vectors')
    
    vectors = []
    for sentence in df['review'].values:
        vectors.append(
            sentence_to_vec(s=sentence,
                            embeddings_dict=embeddings,
                            stop_words=[],
                            tokenizer=word_tokenize
                            )
        )

    vectors = np.array(vectors)
    print(f'Created: {round(time.time()-time_init)} secs.')

    y = df['sentiment'].values  
    
    print('Training folds')
    kf = model_selection.StratifiedKFold(n_splits=5)
    for _fold, (_t, _v) in enumerate(kf.split(X=vectors, y=y)):
        time_init = time.time()
        print(f'Fold={_fold}')
        
        X_train = vectors[_t, :]
        y_train = y[_t]
        
        X_test = vectors[_v, :]
        y_test = y[_v]
        
        model = linear_model.LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        print(f'Accuracy: {accuracy}, Time: {round(time.time()-time_init,2)} secs.\n')