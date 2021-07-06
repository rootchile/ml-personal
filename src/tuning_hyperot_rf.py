import numpy as np
import pandas as pd

from functools import partial

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params, x, y, kfolds=5):
    
    """
    Optimization function
    :param params: dict of params 
    :param x: training data
    :param y: target data
    :return: negative accuracy after k-fold
    """
    
    model = ensemble.RandomForestClassifier(**params)
    
    kf = model_selection.StratifiedKFold(n_splits=kfolds)
    accuracies = []
    
    for idx in kf.split(x, y):
        train_idx, test_idx = idx[0], idx[1]
        
        x_train = x[train_idx]
        y_train = y[train_idx]
        
        x_test = x[test_idx]
        y_test = y[test_idx]
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        fold_accuracy = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(fold_accuracy)
        
    return -1*np.mean(accuracies)
    
if __name__ == '__main__':
    
    df = pd.read_csv('../data/mobile-pricing/train.csv')
    X = df.drop('price_range', axis=1).values
    y = df['price_range'].values
    
    param_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 15, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1500, 1)),
        'criterion': hp.choice('criterio',['gini', 'entropy']),
        'max_features': hp.uniform('max_features', 0, 1)
    }
    
    fitness_function = partial(
        optimize,
        x=X,
        y=y
    )
    
    # keep logging information
    trials = Trials()
    
    hopt = fmin(
        fn = fitness_function,
        space = param_space,
        algo = tpe.suggest,
        max_evals = 15,
        trials = trials
    )
    
    print(hopt)
    
    