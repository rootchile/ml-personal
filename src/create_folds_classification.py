"""
Function to generate stratified k-fold for classification problems
"""
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection

def create_folds_classification(data, n_splits=5):
    
    data['kfold'] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    
    y = data.target.values
    kf = model_selection.StratifiedKFold(n_splits=n_splits)
    
    for f, (_t, _v) in enumerate(kf.split(X = data, y = y)):
        data.loc[_v, 'kfold'] = f
    
    return data

# data = pd.read_csv('../data/mnist/mnist_train.csv')
# data.rename(columns={'label':'target'}, inplace=True)
# data_kfolds = create_folds_classification(data)
# print(data_kfolds.groupby(['kfold'])['target'].count())
#data_kfolds = data_kfolds.to_csv('../data/mnist/mnist_train_folds.csv', index=False)


# data = pd.read_csv('../data/adult-census-income/income.csv')
# data.rename(columns={'income':'target'}, inplace=True)
# data_kfolds = create_folds_classification(data)
# print(data_kfolds.groupby(['kfold'])['target'].count())
# data_kfolds = data_kfolds.to_csv('../data/adult-census-income/income_folds.csv', index=False)