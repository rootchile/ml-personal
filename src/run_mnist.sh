#!/bin/sh
echo 'model|fold|auc_roc|time_sec'
python3 train_mnist.py --fold 0 --model decision_tree_gini
python3 train_mnist.py --fold 1 --model decision_tree_gini
python3 train_mnist.py --fold 2 --model decision_tree_gini
python3 train_mnist.py --fold 3 --model decision_tree_gini
python3 train_mnist.py --fold 4 --model decision_tree_gini

python3 train_mnist.py --fold 0 --model decision_tree_entropy
python3 train_mnist.py --fold 1 --model decision_tree_entropy
python3 train_mnist.py --fold 2 --model decision_tree_entropy
python3 train_mnist.py --fold 3 --model decision_tree_entropy
python3 train_mnist.py --fold 4 --model decision_tree_entropy

python3 train_mnist.py --fold 0 --model random_forest
python3 train_mnist.py --fold 1 --model random_forest
python3 train_mnist.py --fold 2 --model random_forest
python3 train_mnist.py --fold 3 --model random_forest
python3 train_mnist.py --fold 4 --model random_forest