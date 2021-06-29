from sklearn import linear_model
from sklearn import ensemble
from sklearn import preprocessing
import xgboost as xgb

models = {
    'logreg': {
        'model': linear_model.LogisticRegression(),
        'preprocessing':  preprocessing.OneHotEncoder(),
        'preprocessing_by_feature': False,
        'svd_decomposition': False,
        'svd_components': 0,
    },
    'random_forest': {
        'model': ensemble.RandomForestClassifier(n_jobs=-1),
        'preprocessing': preprocessing.LabelEncoder(),
        'preprocessing_by_feature': True,
        'svd_decomposition': False,
        'svd_components': 0,
    }
    ,
    'random_forest_svd': {
        'model': ensemble.RandomForestClassifier(n_jobs=-1),
        'preprocessing': preprocessing.OneHotEncoder(),
        'preprocessing_by_feature': False,
        'svd_decomposition': True,
        'svd_components': 120,
    }
    ,
    'xgboost_classifier': {
        'model': xgb.XGBClassifier(n_jobs=-1,  verbosity=0, max_depth=7, n_estimators=200),
        'preprocessing': preprocessing.LabelEncoder(),
        'preprocessing_by_feature': True,
        'svd_decomposition': False,
        'svd_components': 0,
    }
}