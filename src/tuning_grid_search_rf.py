import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

if __name__ == '__main__':
    
    df = pd.read_csv('../data/mobile-pricing/train.csv')
    X = df.drop('price_range', axis=1).values
    y = df['price_range'].values
    
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    
    param_grid = {
        'n_estimators': [100, 200, 250, 300, 400, 500],
        'max_depth': [1, 2, 5, 7, 11, 15],
        'criterion': ['gini', 'entropy']
    }
    
    model = model_selection.GridSearchCV(estimator=classifier,
                                         param_grid=param_grid,
                                         scoring='accuracy',
                                         verbose=1,
                                         n_jobs=-1,
                                         cv=5)
    model.fit(X, y)
    print('Best score: {}'.format(model.best_score_))
    print('Best parameters set:')
    best_params = model.best_estimator_.get_params()
    for param in sorted(param_grid.keys()):
        print(f'\t{param}: {best_params[param]}')
        