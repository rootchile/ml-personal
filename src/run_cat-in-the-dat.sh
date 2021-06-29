#!/bin/sh
echo 'model|fold|auc_roc|time_sec'
python3 -W ignore train_catinthedat.py --fold 0 --model logreg 
python3 -W ignore train_catinthedat.py --fold 1 --model logreg 
python3 -W ignore train_catinthedat.py --fold 2 --model logreg 
python3 -W ignore train_catinthedat.py --fold 3 --model logreg 
python3 -W ignore train_catinthedat.py --fold 4 --model logreg 

python3 -W ignore train_catinthedat.py --fold 0 --model random_forest 
python3 -W ignore train_catinthedat.py --fold 1 --model random_forest 
python3 -W ignore train_catinthedat.py --fold 2 --model random_forest 
python3 -W ignore train_catinthedat.py --fold 3 --model random_forest 
python3 -W ignore train_catinthedat.py --fold 4 --model random_forest 

python3 -W ignore train_catinthedat.py --fold 0 --model random_forest_svd
python3 -W ignore train_catinthedat.py --fold 1 --model random_forest_svd
python3 -W ignore train_catinthedat.py --fold 2 --model random_forest_svd
python3 -W ignore train_catinthedat.py --fold 3 --model random_forest_svd
python3 -W ignore train_catinthedat.py --fold 4 --model random_forest_svd

python3 -W ignore train_catinthedat.py --fold 0 --model xgboost_classifier
python3 -W ignore train_catinthedat.py --fold 1 --model xgboost_classifier
python3 -W ignore train_catinthedat.py --fold 2 --model xgboost_classifier
python3 -W ignore train_catinthedat.py --fold 3 --model xgboost_classifier
python3 -W ignore train_catinthedat.py --fold 4 --model xgboost_classifier