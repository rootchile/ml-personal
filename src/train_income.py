import os
import argparse
import time
import itertools

import joblib
import pandas as pd
from sklearn import metrics

import config_income as config
import model_dispatcher_income as model_dispatcher

def feature_engineering_cat2d(df, cat_cols):
    """
    This function create all combinations of degree two for categorical columns
    
    :param df: pandas dataframe with train+validate+test data
    :param cat_cols: list of categorical columns
    :return: pandas dataframe with news features
    """
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1+'_'+c2] = df[c1].astype(str)+ '_' + df[c2].astype(str)
        
    return df
    
def run(fold, model, target='target'):
    
    time_init = time.time()
    df = pd.read_csv(config.TRAINING_FILE)
    method_preprocessing = model_dispatcher.models[model]['preprocessing'] 
    method_by_feature = model_dispatcher.models[model]['preprocessing_by_feature'] 
    numerical_features = model_dispatcher.models[model]['numerical_features']
    categorical_features_d2 = model_dispatcher.models[model]['categorical_features_d2']

    numerical_cols = ['fnlwgt', 'age', 'capital.gain', 'capital.loss','hours.per.week']
    categorical_cols = [c for c in df.columns 
                            if c not in numerical_cols and c not in ['kfold',target]]
    
    if not numerical_features:
        # drop numerical features
        df.drop(numerical_cols, axis=1, inplace=True)
    
    target_mapping = {
        '<=50K': 0,
        '>50K': 1
    }
    
    df.loc[:, 'target'] = df['target'].map(target_mapping)
    
    if categorical_features_d2:
        df = feature_engineering_cat2d(df, categorical_cols)
    
    features = [ f for f in df.columns if f not in ['kfold', target]]
    
    for col in features:
        if col not in numerical_cols:
            df.loc[:,col] = df.loc[:,col].astype(str).fillna('NONE')
        

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    x_train = []
    x_valid = []

    df_full = pd.concat(
                        [df_train[features], df_valid[features]]
                        , axis=0) #by index    
    # Ex. OneHotEncoder
    if not method_by_feature:
        preproc = method_preprocessing
        preproc.fit(df_full)
        x_train = preproc.transform(df_train[features])
        x_valid = preproc.transform(df_valid[features])

    else:
        # Ex. LabelEncoder
        for col in features:
            if col not in numerical_cols:
                preproc = method_preprocessing
                preproc.fit(df_full[col])
                df_train[col] = preproc.transform(df_train[col])
                df_valid[col] = preproc.transform(df_valid[col])

        x_train = df_train[features].values
        x_valid = df_valid[features].values
    
    y_train = df_train[target].values
    y_valid = df_valid[target].values
 
  
    clf = model_dispatcher.models[model]['model']
    clf.fit(x_train, y_train)
    
    y_prob = clf.predict_proba(x_valid)[:,1]
    roc_auc_score = metrics.roc_auc_score(y_valid, y_prob)
    
    time_sec = time.time()-time_init
    print(f'{model}|{fold}|{round(roc_auc_score,4)}|{round(time_sec,2)}')
    
    # persist model
    joblib.dump(clf, 
                os.path.join(config.MODEL_OUTPUT,f'{model}_{fold}.bin')
                )
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',type=int)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    
    run(fold = args.fold,
        model = args.model)