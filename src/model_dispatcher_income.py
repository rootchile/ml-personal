from sklearn import linear_model
from sklearn import ensemble
from sklearn import preprocessing
import xgboost as xgb

"""
numerical_features: 1 if drop numerical features from dataset, else 0.
categorical_feature_2d: 1 if use feature engineering (combinations degree two of categorical features), else 0.
"""
models = {
    'logreg': {
        'model': linear_model.LogisticRegression(),
        'preprocessing':  preprocessing.OneHotEncoder(),
        'preprocessing_by_feature': False,
        'numerical_features': False,
        'categorical_features_d2': False,
        'target_encoding': False

    },
    'xgboost_classifier': {
        'model': xgb.XGBClassifier(n_jobs=-1,  verbosity=0, max_depth=7, n_estimators=200),
        'preprocessing':  preprocessing.LabelEncoder(),
        'preprocessing_by_feature': True,
        'numerical_features': False,
        'categorical_features_d2': False,
        'target_encoding': False

    },
    'xgboost_classifier_with_numeric': {
        'model': xgb.XGBClassifier(n_jobs=-1, verbosity=0, max_depth=7, n_estimators=200),
        'preprocessing':  preprocessing.LabelEncoder(),
        'preprocessing_by_feature': True,
        'numerical_features': True,
        'categorical_features_d2': False,
        'target_encoding': False

    },
    
    'xgboost_classifier_with_features2d': {
        'model': xgb.XGBClassifier(n_jobs=-1,  verbosity=0, max_depth=7, n_estimators=200),
        'preprocessing':  preprocessing.LabelEncoder(),
        'preprocessing_by_feature': True,
        'numerical_features': True,
        'categorical_features_d2': True,
        'target_encoding': False

    },
    
        
    'xgboost_classifier_with_targetencoding': {
        'model': xgb.XGBClassifier(n_jobs=-1,  verbosity=2, max_depth=7, n_estimators=200, ),
        'preprocessing':  preprocessing.LabelEncoder(),
        'preprocessing_by_feature': True,
        'numerical_features': False,
        'categorical_features_d2': False,
        # 'target_encoding': True
    },
}