import numpy as np
import pandas as pd

from functools import partial 

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection


from skopt import gp_minimize
from skopt import space

def optimize(params, param_names, x, y, kfolds=5):
    
    """
    Optimization function 
    
    :param params: list of params from gp_minimize
    :param param_names: list of param name, order is important.
    :param x: training data
    :param y: target data
    :param kfolds: integer, number of folds (default=5)
    
    :return: negative accuracy after k-folds
    """
    
    params = dict(zip(param_names, params))
    
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
    
    param_space = [
        # max_depth
        space.Integer(3,15, name='max_depth'),
        # n estimators
        space.Integer(100,1500, name='n_estimators'),
        # criterion is a categorical 
        space.Categorical(['gini', 'entropy'], name='criterion'),
        # distribution
        space.Real(0.01, 1, prior='uniform', name='max_features')
    ]
     
    param_names = ['max_depth', 'n_estimators', 'criterion', 'max_features']

    fitness_function = partial(optimize,
                                param_names=param_names, 
                                x=X, 
                                y=y)
    
    result = gp_minimize(
            fitness_function,
            dimensions=param_space,
            n_calls=15,
            n_random_starts=10,
            verbose=True
    )
    
    best_params = dict(zip(param_names, result.x))
    
    print(best_params)