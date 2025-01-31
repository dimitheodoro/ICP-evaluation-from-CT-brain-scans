import numpy as np
from scipy import stats

def auc_from_confusion_matrix(tp, tn, fp, fn):
    """Calculate AUC from confusion matrix elements"""
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    # AUC approximation using trapezoid rule
    return (sensitivity + specificity) / 2

def auc_ci_confusion_matrix(tp, tn, fp, fn, n_bootstraps=1000, confidence_level=0.95):
    bootstrapped_scores = []
    total_pos = tp + fn
    total_neg = tn + fp
    
    for _ in range(n_bootstraps):
        # Bootstrap positive cases
        bootstrap_tp = np.random.binomial(total_pos, tp/(tp + fn))
        bootstrap_fn = total_pos - bootstrap_tp
        
        # Bootstrap negative cases
        bootstrap_tn = np.random.binomial(total_neg, tn/(tn + fp))
        bootstrap_fp = total_neg - bootstrap_tn
        
        # Calculate AUC for this bootstrap sample
        auc = auc_from_confusion_matrix(
            bootstrap_tp, bootstrap_tn, 
            bootstrap_fp, bootstrap_fn
        )
        bootstrapped_scores.append(auc)
    
    # Calculate confidence intervals
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(bootstrapped_scores, alpha * 100)
    ci_upper = np.percentile(bootstrapped_scores, (1 - alpha) * 100)
    
    # Calculate mean AUC
    mean_auc = auc_from_confusion_matrix(tp, tn, fp, fn)
    
    return mean_auc, ci_lower, ci_upper

# 3D custom 
# tn: 79, fp: 37, fn: 8, tp: 32

# DenseNet
# tn: 69, fp: 47, fn: 4, tp: 36

# ResNet
# tn: 116, fp: 0, fn: 40, tp: 0

# MobilNet
# tn: 68, fp: 48, fn: 1, tp: 39

architectures = ["3D custom", "DenseNet","ResNet","MobilNet"] 
tp = [32, 36, 0, 39]
tn = [79, 69, 116, 68]
fp = [37, 47, 0, 48]
fn = [8, 4, 40, 1]

for i in range(4):
    mean_auc, ci_lower, ci_upper = auc_ci_confusion_matrix(tp[i], tn[i], fp[i], fn[i])
    print(f"{architectures[i]} AUC: {mean_auc:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")

"""
3D custom AUC: 0.741 (95% CI: 0.672-0.816)
DenseNet AUC: 0.747 (95% CI: 0.679-0.811)
ResNet AUC: 0.500 (95% CI: 0.500-0.500)
MobilNet AUC: 0.781 (95% CI: 0.729-0.828)
"""