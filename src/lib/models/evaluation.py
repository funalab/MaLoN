##########################################
# Project: Evaluate inferred results
# Author: Yusuke Hiki
##########################################

# MODULE
import configparser
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# MAIN
if __name__ == "__main__":
    PRED_EDGELIST_FILE = sys.argv[1]
    TRUE_EDGELIST_FILE = sys.argv[2]
    DATASET_NAME = sys.argv[3]
    OUTPUT_DIR = sys.argv[4]

    # Load predicted and true edgelist
    pred_edgelist = pd.read_csv(PRED_EDGELIST_FILE, sep='\t')
    true_edgelist = pd.read_csv(TRUE_EDGELIST_FILE, sep='\t', header=None)
    if true_edgelist.shape[1] >= 3:
        true_edgelist = true_edgelist.iloc[np.array(true_edgelist.iloc[:, 2] == 1), :2]

    # Get predicted score and true label
    score, true = [], []
    for i_edge in range(len(pred_edgelist.index)):
        # Get source gene, target gene, and score of target edge
        source_i = pred_edgelist.iloc[i_edge, 0]
        target_i = pred_edgelist.iloc[i_edge, 1]
        score_i = pred_edgelist.iloc[i_edge, 2]

        # Get true label
        index_i = true_edgelist.iloc[:, 0] + true_edgelist.iloc[:, 1] == source_i + target_i
        if any(index_i):  # Target edge exists in true edgelist: add true label
            true_i = 1
        else:  # Target edge doesn't exists in true edgelist (ex. edgelist includes only true edge): add false label
            true_i = 0

        # Append score and true label of target edge
        score.append(score_i)
        true.append(true_i)
        
    # Calculate FPR, TPR, thresholds, and AUROC
    fpr, tpr, thresholds = roc_curve(true, score)
    auroc = roc_auc_score(true, score)

    # Add directory if not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Plot ROC curve
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.5f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='dotted')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve (' + DATASET_NAME + ')')
    plt.grid(True)
    plt.legend(loc="lower right")
    pp1 = PdfPages(OUTPUT_DIR + '/' + DATASET_NAME + "_roc_curve.pdf")
    pp1.savefig()
    pp1.close()
    plt.close()

    # Calculate precision, recall, thresholds, and AUPR
    precision, recall, thresholds = precision_recall_curve(true, score)
    aupr = average_precision_score(true, score)

    # Plot PR curve
    plt.plot(recall, precision, color='red', lw=2, label='PR curve (area = %0.5f)' % aupr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve (' + DATASET_NAME + ')')
    plt.grid(True)
    plt.legend(loc="upper right")
    pp1 = PdfPages(OUTPUT_DIR + '/' + DATASET_NAME + "_pr_curve.pdf")
    pp1.savefig()
    pp1.close()
    plt.close()

    # Print AUROC and AUPR
    print(DATASET_NAME + "\n  AUROC: " + str(auroc) + "\n  AUPR: " + str(aupr))
