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
    
    params = {
        'n_estimators': np.arange(100,1500,100),
        'max_depth': np.arange(1,31),
        'criterion': ['gini', 'entropy']
    }
    
    model = model_selection.RandomizedSearchCV(estimator=classifier,
                                         param_distributions=params,
                                         scoring='accuracy',
                                         verbose=1,
                                         n_jobs=-1,
                                         cv=5)
    model.fit(X, y)
    print('Best score: {}'.format(model.best_score_))
    print('Best parameters set:')
    best_params = model.best_estimator_.get_params()
    for param in sorted(params.keys()):
        print(f'\t{param}: {best_params[param]}')
        