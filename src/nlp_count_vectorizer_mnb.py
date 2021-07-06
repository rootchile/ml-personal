import time

import pandas as pd
import nltk
nltk.data.path.append('../nltk_data/')

from nltk.tokenize import word_tokenize
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
if __name__ == '__main__':
    
    df = pd.read_csv('../data/imdb/imdb.csv')
    
    # target
    df.sentiment = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    df['kfold'] = -1
    # randomize
    df = df.sample(frac=1).reset_index(drop=True)
    
    y = df['sentiment'].values
    
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    for f, (_t, _v) in enumerate(kf.split(X=df, y=y)):
        df.loc[_v, 'kfold'] = f
    
    time_init = time.time()
    for _fold in range(5):
        df_train = df[df.kfold != _fold].reset_index(drop=True)
        df_test  = df[df.kfold == _fold].reset_index(drop=True)
        
        count_vectorizer = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)
        count_vectorizer.fit(df_train.review)
        
        x_train = count_vectorizer.transform(df_train.review)
        x_test  = count_vectorizer.transform(df_test.review)
        
        y_train = df_train.sentiment.values
        y_test  = df_test.sentiment.values
        
        model = naive_bayes.MultinomialNB()
        model.fit(x_train, y_train)
        
        y_pred = model.predict(x_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        time_sec = time.time()-time_init
        #model|fold|accuracy|time_sec
        print(f'multinomialNB|{_fold}|{accuracy}|{time_sec}')