import os
import argparse
import time

import pandas as pd
from sklearn import metrics
from sklearn import decomposition
import joblib

from scipy import sparse

import config_catinthedat as config
import model_dispatcher_cat as model_dispatcher

def run(fold, model, target='target'):
    
    time_init = time.time()
    df = pd.read_csv(config.TRAINING_FILE)
    
    features = [ f for f in df.columns if f not in ['id','kfold', target]]
    
    for col in features:
        df.loc[:,col] = df.loc[:,col].astype(str).fillna('NONE')
        
    
    method_preprocessing = model_dispatcher.models[model]['preprocessing'] 
    method_by_feature = model_dispatcher.models[model]['preprocessing_by_feature'] 
    svd_decomposition = model_dispatcher.models[model]['svd_decomposition']
    svd_components_n =   model_dispatcher.models[model]['svd_components']

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
            preproc = method_preprocessing
            preproc.fit(df_full[col])
            df_train[col] = preproc.transform(df_train[col])
            df_valid[col] = preproc.transform(df_valid[col])

        x_train = df_train[features].values
        x_valid = df_valid[features].values
    
    y_train = df_train[target].values
    y_valid = df_valid[target].values
 
    if svd_decomposition:
        svd = decomposition.TruncatedSVD(n_components=svd_components_n)
        full_sparse = sparse.vstack((x_train, x_valid))
        svd.fit(full_sparse)
        
        x_train = svd.transform(x_train)
        x_valid = svd.transform(x_valid)
        
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