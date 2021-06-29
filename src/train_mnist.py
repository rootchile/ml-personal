import os
import argparse
import time

import joblib
import pandas as pd
from sklearn import metrics

import config_mnist as config
import model_dispatcher_mnist as model_dispatcher

def run(fold, model, target='target'):
    time_init = time.time()

    df = pd.read_csv(config.TRAINING_FILE)
    # exclude this fold for training
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    # validate with this one
    df_valid = df[df.kfold==fold].reset_index(drop=True)
    
    # numpy arrays without label column
    x_train = df_train.drop(target,axis=1).values
    y_train = df_train[target].values
    
    x_valid = df_valid.drop(target,axis=1).values
    y_valid = df_valid['target'].values
    
    # simple decision tree
    clf = model_dispatcher.models[model]
    clf = clf.fit(x_train, y_train)
    
    # predictions for validation
    y_pred = clf.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, y_pred)
   
    time_sec = time.time()-time_init
    print(f'{model}|{fold}|{round(accuracy,4)}|{round(time_sec,2)}')
    
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