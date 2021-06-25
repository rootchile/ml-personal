
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from metrics import tpr, fpr
tpr_list = []
fpr_list = []

# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]

# prob of a sample being 1 
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

for thresh in thresholds:
    
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    temp_tpr = tpr(y_true, temp_pred)
    temp_fpr = fpr(y_true, temp_pred)
    
    tpr_list.append(temp_tpr) 
    fpr_list.append(temp_fpr)

# print('Thresh\tTPR\tFPR')
# for t, tpr, fpr in zip(thresholds, tpr_list, fpr_list):
#     print(f'{t}\t{tpr}\t{fpr}')
auc = roc_auc_score(y_true, y_pred)

plt.figure(figsize=(7,7))
plt.fill_between(fpr_list, tpr_list, alpha=0.3)
plt.plot(fpr_list, tpr_list,lw=3)
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'AUC = {auc}')
plt.savefig('./reports/example_roc.png')