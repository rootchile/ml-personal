import os
import gc
import config_catinthedat as config

import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils


def create_model(data, cat_columns):
    """
    Create a compiled tf.keras model for entity embeddings.
    """
    
    inputs = []
    outputs = []
    
    for col in cat_columns:
        uniques_values = int(data[col].nunique())
        embed_dimension = int(min(np.ceil(uniques_values/2), 50))
        inp = layers.Input(shape=(1,))
        
        out = layers.Embedding(uniques_values+1, embed_dimension, name=col)(inp)
        out = layers.SpatialDropout1D(0.3)(out)

        out = layers.Reshape(target_shape=(embed_dimension, ))(out)
        
        inputs.append(inp)
        outputs.append(out)
    
    x = layers.Concatenate()(outputs)
    
    x = layers.BatchNormalization()(x)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    y = layers.Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=y)
    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    return model

def run(fold, target='target'):
    
    df = pd.read_csv(config.TRAINING_FILE)
    
    features = [ f for f in df.columns if f not in ['id','kfold', target]]
    
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')
    
    for col in features:
        encoder = preprocessing.LabelEncoder()
        df.loc[:, col] = encoder.fit_transform(df[col].values)
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True) 
    
    model = create_model(df, features)
    
    x_train = [ df_train[features].values[:, k] for k in range(len(features))]
    x_valid = [ df_valid[features].values[:, k] for k in range(len(features))]    
    
    y_train = df_train[target].values
    y_valid = df_valid[target].values

    #target binarization
    y_train_cat = utils.to_categorical(y_train)
    y_valid_cat = utils.to_categorical(y_valid)
    
    model.fit(x = x_train,
              y = y_train_cat,
              validation_data=(x_valid, y_valid_cat),
              verbose=0,
              batch_size=1024,
              epochs=3
             )
    
    # validation
    y_preds = model.predict(x_valid)[:, 1]
    auc_roc = metrics.roc_auc_score(y_valid, y_preds)
    print(f'AUC ROC: {auc_roc}')
    
    K.clear_session()
    
if __name__ == '__main__':
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)