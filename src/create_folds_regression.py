"""
Function to generate stratified k-fold for regression
"""
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds_regression(data, n_splits=5):
    
    data['kfold'] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Sturge's rule
    num_bins = int(np.floor(1+np.log2(len(data))))
    
    data.loc[:, "bins"] = pd.cut(   
                                data['target'],
                                bins= num_bins, 
                                labels=False
                        )
    
    kf = model_selection.StratifiedKFold(n_splits=n_splits)
    
    for f, (_t, _v) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[_v, 'kfold'] = f

    data.drop('bins', axis=1, inplace=True)
    
    return data

data = pd.read_csv('../data/cat-in-the-dat/cat_train.csv')
data_kfolds = create_folds_regression(data)
print(data_kfolds.groupby(['kfold'])['target'].count())
data_kfolds.to_csv('../data/cat-in-the-dat/cat_train_folds.csv', index=False)