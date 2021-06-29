#!/bin/sh
echo 'model|fold|auc_roc|time_sec'
python3 -W ignore train_income.py --fold 0 --model logreg 
python3 -W ignore train_income.py --fold 1 --model logreg 
python3 -W ignore train_income.py --fold 2 --model logreg 
python3 -W ignore train_income.py --fold 3 --model logreg 
python3 -W ignore train_income.py --fold 4 --model logreg 

python3 -W ignore train_income.py --fold 0 --model xgboost_classifier 
python3 -W ignore train_income.py --fold 1 --model xgboost_classifier 
python3 -W ignore train_income.py --fold 2 --model xgboost_classifier 
python3 -W ignore train_income.py --fold 3 --model xgboost_classifier 
python3 -W ignore train_income.py --fold 4 --model xgboost_classifier 

python3 -W ignore train_income.py --fold 0 --model xgboost_classifier_with_numeric
python3 -W ignore train_income.py --fold 1 --model xgboost_classifier_with_numeric
python3 -W ignore train_income.py --fold 2 --model xgboost_classifier_with_numeric
python3 -W ignore train_income.py --fold 3 --model xgboost_classifier_with_numeric
python3 -W ignore train_income.py --fold 4 --model xgboost_classifier_with_numeric

python3 -W ignore train_income.py --fold 0 --model xgboost_classifier_with_features2d
python3 -W ignore train_income.py --fold 1 --model xgboost_classifier_with_features2d
python3 -W ignore train_income.py --fold 2 --model xgboost_classifier_with_features2d
python3 -W ignore train_income.py --fold 3 --model xgboost_classifier_with_features2d
python3 -W ignore train_income.py --fold 4 --model xgboost_classifier_with_features2d