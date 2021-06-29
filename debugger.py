from metrics import metrics

if __name__ == '__main__':
  
    l1 = [0,1,1,1,0,0,0,1]
    l2 = [0,1,0,1,0,1,0,0]

    accuracy = metrics.accuracy(l1,l2) 
    accuracy2= metrics.accuracy_v2(l1,l2)

    tp = metrics.true_positive(l1,l2)
    tn = metrics.true_negative(l1,l2)
    fp = metrics.false_positive(l1,l2)
    fn = metrics.false_negative(l1,l2)

    precision = metrics.precision(l1,l2)
    recall = metrics.recall(l1,l2)
    f1 = metrics.f1(l1,l2)
    specificity = metrics.specificity(l1,l2)

    print(f'Accuracy {accuracy}')
    print(f'TP {tp}')
    print(f'TN {tn}')
    print(f'FP {fp}')
    print(f'FN {fn}')
    print(f'Precision {precision}')
    print(f'Recall / True Positive Rate / Sensitivity: {recall}')
    print(f'F1 {f1}')
    print(f'Specificity / True Negative Rate:  {specificity}')



    y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    y_prob = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

    log_loss = metrics.log_loss(y_true,y_prob)

    from sklearn.metrics import log_loss as sklearn_logloss
    log_loss_sk = sklearn_logloss(y_true,y_prob)

    print(f'Log Loss {log_loss}')
    print(f'Log Loss sklearn {log_loss_sk}')


    y_true_multiclass = [0,1,2,0,1,2,0,2,2]
    y_pred_multiclass = [0,2,1,0,2,1,0,0,2]

    macro_p = metrics.macro_precision(y_true_multiclass, y_pred_multiclass)
    print(f'Macro Precision  {macro_p}')

    micro_p = metrics.micro_precision(y_true_multiclass, y_pred_multiclass)
    print(f'Micro Precision  {micro_p}')


    weighted_p = metrics.weighted_precision(y_true_multiclass, y_pred_multiclass)
    print(f'Weighted Precision  {weighted_p}')
    
    weighted_f1 = metrics.weighted_f1(y_true_multiclass, y_pred_multiclass)
    print(f'Weighted F1  {weighted_f1}')
    
      
    mae = metrics.mae(l1, l2)
    print(f'MAE {mae}')
    
    mse = metrics.mse(l1, l2)
    print(f'MSE {mse}')
    
    rmse = metrics.rmse(l1, l2)
    print(f'RMSE {rmse}')
    
    # msle = metrics.msle(l1,l2)
    # print(f'MSLE {msle}')
    
    # mpe = metrics.mpe(l1,l2)
    # print(f'MPE {mpe}')
    
    # mape = metrics.mape(l1,l2)
    # print(f'MAPE {mape}')
    
    r2 = metrics.r2(l1,l2)
    print(f'R^2 {r2}')