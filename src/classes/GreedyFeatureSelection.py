import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification

class GreedyFeatureSelection:
    
    def evaluate_score(self, X, y):
        """
        This function evaluate model on daa and returns Area Under ROC Curve (AUC)
        
        :param X: training data
        :param y: targets
        :return: AUC ROC
        """
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        y_pred = model.predict_proba(X)[:, 1]
        auc_roc = metrics.roc_auc_score(y, y_pred)
        return auc_roc
        
    def _feature_selection(self, X, y):
        """
        Greedy selection
        
        :param X: data (numpy array)
        :param y: targets (numpy array)
        
        :return: (best scores, best features)
        """
        
        best_features = []
        best_scores = []
        
        n_features = X.shape[1] # (row,col)
        
        i = 0
        while self.max_iter > i:
            
            this_feature = None
            best_score = 0
            
            for f in range(n_features):
                
                # skip
                if f in best_features:
                    continue
            
                selected_features = best_features + [f]

                X_train = X[:, selected_features]
                score = self.evaluate_score(X_train, y)
                
                if score > best_score:
                    this_feature = f
                    best_score = score
            
            if this_feature != None:
                best_features.append(this_feature)
                best_scores.append(best_score)
                
            # if we didnt improve during the previous round
            if len(best_scores) > 2:
                if best_scores[-1] < best_scores[-2]:
                    break
            i+=1
        return best_scores[:-1], best_features[:-1]
    
    def __call__(self, X, y, max_iter=5000):
        self.max_iter = max_iter
        scores, features = self._feature_selection(X, y)
        return X[:, features], scores 
        
