"""
Multiple metrics use for ML algorithms from scratch
root.chile@gmail.com 
"""
import numpy as np
from collections import Counter


def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return accuracy score
    """
    
    correct_counter = 0
    
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1
    return correct_counter / len(y_true)


def accuracy_v2(y_true, y_pred):
    """
    Function to calculate Accuracy from TP,TN,FP,FN
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return accuracy score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    
    accuracy_score = (tp+tn)/(tp+tn+fp+fn)
    return accuracy_score
    
def true_positive(y_true, y_pred):
    """
    Function to calculate True Positives 
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true positive values
    """
    
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    
    return tp

def true_negative(y_true, y_pred):
    """
    Function to calculate True Negatives 
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true negatives values
    """
    
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    
    return tn


def false_positive(y_true, y_pred):
    """
    Function to calculate False Positive 
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false positive values
    """
    
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    
    return fp

def false_negative(y_true, y_pred):
    """
    Function to calculate False Negative 
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false negative values
    """
    
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    
    return fn


def precision(y_true, y_pred):
    """
    Function to calculate precision
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: precision score
    """
    
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    
    precision_score = tp/(tp+fp)
    
    return precision_score

def recall(y_true, y_pred):
    """
    Function to calculate recall
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: recall score
    """
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall_score = tp/(tp+fn)
    
    return recall_score

def tpr(y_true, y_pred):
    """
    Function to calculate True Positive Rate 
    """
    return recall(y_true, y_pred)

def sensitivity(y_true, y_pred):
    """
    Function to calculate Sensitivity
    """
    return recall(y_true, y_pred)


def fpr(y_true, y_pred):
    """
    Function to calculate False Positive Rate
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: false positive rate
    """
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return fp/(tn+fp)

def specificity(y_true, y_pred):
    """
    Function to calculate Specificity
    """
    return (1-fpr(y_true, y_pred))

def tnr(y_true, y_pred):
    """
    Function to calculate True Negative Rate
    """
    return (1-fpr(y_true, y_pred))

def f1(y_true, y_pred):
    """
    Function to calculate f1 score
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: f1 score
    """
    
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    
    f1_score = 2 * precision_score * recall_score / (precision_score+recall_score)
    
    return f1_score

def log_loss(y_true, y_prob, epsilon = 1e-15):
    """
    Function to calculate log logss
    
    :param y_true: list of true values
    :param y_prob: list of probabilities for 1
    :return: overall log loss
    """
    
    loss = []
    
    for yt, yp in zip(y_true, y_prob):
        yp = np.clip(yp, epsilon, 1-epsilon)
        temp_loss = -1.0 * (yt*np.log(yp) + (1-yt)*np.log(1-yp))
        loss.append(temp_loss)
        
    return np.mean(loss)
    
    
def macro_precision(y_true,y_pred):
    """
    Function to calculate macro averaged precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: macro precision score
    """
    
    num_classes = len(np.unique(y_true))
    precision = 0
    
    for _class in range(num_classes):
        
        #all class except current are considered negative
        tmp_true = [ 1 if p == _class else 0 for p in y_true]
        tmp_pred = [ 1 if p == _class else 0 for p in y_pred]
        
        tp = true_positive(tmp_true,tmp_pred)
        fp = false_positive(tmp_true,tmp_pred)
        tmp_precision = tp / (tp+fp)
        
        precision += tmp_precision
    
    precision /= num_classes
    
    return precision

def micro_precision(y_true, y_pred):
    """
    Function to calculate micro average precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: micro precision score
    """
    
    num_classes = len(np.unique(y_true))
    
    tp = 0
    fp = 0
    
    for _class in range(num_classes):
        tmp_true = [ 1 if p == _class else 0 for p in y_true]
        tmp_pred = [ 1 if p == _class else 0 for p in y_pred]
        
        tp += true_positive(tmp_true, tmp_pred)
        fp += false_positive(tmp_true, tmp_pred)
    
    precision = tp / (tp +fp)
    
    return precision

def weighted_precision(y_true, y_pred):
    """
    Function to calculate weighted average precision
    
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: weighted precision score
    """
    
    num_classes = len(np.unique(y_true))
    # dict classes {0: n, 1: m, 2:..}
    class_counts = Counter(y_true)
    
    precision = 0
    
    for _class in range(num_classes):
        tmp_true = [ 1 if p == _class else 0 for p in y_true]
        tmp_pred = [ 1 if p == _class else 0 for p in y_pred]
        
        tp = true_positive(tmp_true, tmp_pred)
        fp = false_positive(tmp_true, tmp_pred)
        
        tmp_precision = tp/(tp+fp)
        
        weighted_precision = class_counts[_class] * tmp_precision
        
        precision += weighted_precision
        
    overall_precision = precision / len(y_true)
    
    return overall_precision

def weighted_f1(y_true, y_pred):
    """
    Function to calculate weighted f1 score
    
    :param y_true: list of true values
    :param y_prob: list of predicted probabilities
    
    :return:  weigthed f1 score
    """
    
    num_classes = len(np.unique(y_true))
    # dict of classes. {0: n, 1:m, 2:...}
    class_counts = Counter(y_true)
    
    f1 = 0
    for _class in range(num_classes):
        tmp_true = [ 1 if p == _class else 0 for p in y_true]
        tmp_pred = [ 1 if p == _class else 0 for p in y_pred]
        
        p = precision(tmp_true, tmp_pred)
        r = recall(tmp_true, tmp_pred)
        
        if (p + r) != 0:
            tmp_f1 = 2 * p * r / (p+r)
        else:
            tmp_f1 = 0
        
        weighted_f1 = class_counts[_class]*tmp_f1
        
        f1 += weighted_f1
    
    overall_f1 = f1 / len(y_true)
    
    return overall_f1


def mae(y_true,y_pred):
    """
    Function to calculate MAE: Mean Absolute Error
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: mae
    """
    
    error = 0
    for yt,yp in zip(y_true,y_pred):
        error += np.abs(yt-yp)
    
    return error/len(y_true)


def mse(y_true,y_pred):
    """
    Function to calculate MSE: Mean Squared Error
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: mse
    """ 
    
    error = 0
    for yt, yp in zip(y_true,y_pred):
        error += np.abs(yt-yp)**2
    
    return error/len(y_true)

def rmse(y_true,y_pred):
    return np.sqrt(mse(y_true,y_pred))


def msle(y_true,y_pred):
    """
    Function to calculate MSLE: Mean Squared Log Error
    :param y_true: list of true values
    :param y_pred: list of predicted values
    
    :return: mean squared logarithmic error
    """
    
    error = 0
    for yt,yp in zip(y_true,y_pred):
        error += (np.log(1+yt)-np.log(1+yp))**2
    
    return error/len(y_true)

def mpe(y_true,y_pred):
    try:
        """
        Function to calculate MPE: Mean Percentage Error
        :param y_true: list of true values
        :param y_pred: list of predicted values
        
        :return: mean percentage error
        """
        
        error = 0
        for yt,yp in zip(y_true,y_pred):
            error += (yt - yp)/yt
        
        return error/len(y_true)
    except:
        return None
    
def mape(y_true,y_pred):
    try:
        """
        Function to calculate MAPE: Mean Absolute Percentage Error
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: mean absolute percentage error
        """

        error = 0
        for yt, yp in zip(y_true,y_pred):
            error += np.abs(yt-yp)/yt
            
        return error/len(y_true)
    except:
        return None

def r2(y_true,y_pred):
    try:
        """
        Function to calculate r-squared score (coefficiente of determination)
        
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: coefficient of determination or r-squared
        
        """
        
        mean_yt= np.mean(y_true)
        numerator = 0
        denominator = 0
        
        for yt, yp in zip(y_true, y_pred):
            numerator += (yt-yp) ** 2
            denominator += (yt-mean_yt) ** 2
        
        ratio = numerator/denominator
        return 1-ratio
    except:
        return None