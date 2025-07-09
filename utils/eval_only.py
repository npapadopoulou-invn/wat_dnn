import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import pandas as pd
# Classification report      
def plot_classification_report_with_support(report, threshold, name):
    labels = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    metrics = ['precision', 'recall', 'f1-score', 'support']
    data = np.array([[report[label][metric] for metric in metrics] for label in labels])
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.matshow(data, cmap='coolwarm')
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(cax)
    # Adding the text
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.title(f'Classification Report with Support for threshold: {threshold}')
    plt.savefig(f"/home/npapadopoulou/wat/{name}/cm/classification_report_th{threshold}.png")
import numpy as np
import matplotlib.pyplot as plt

# Confusion matrix
def cm(name, threshold, label_list, labels):
    os.makedirs(f"/home/npapadopoulou/wat/{name}/cm", exist_ok=True)
    
    y_pred_probs = np.load(f"/home/npapadopoulou/wat/eval_mltools/{name}/y_pred.npy")
    y_test = np.load(f"/home/npapadopoulou/wat/eval_mltools/{name}/y_test.npy")

    y_pred = np.argmax(y_pred_probs, axis=-1)  #(N, 1600)
    y_pred_max_probs = np.max(y_pred_probs, axis=-1)    
    y_pred = np.where(y_pred_max_probs >= threshold, y_pred, 0) # Where the model is NOT confident enough, I will put it as "NONE"

    y_pred_flat = y_pred.flatten()  #
    y_test_flat = y_test.flatten()

    # report = classification_report(y_test_flat, y_pred_flat, target_names=label_list, labels=labels, zero_division=0, output_dict=True)
    # plot_classification_report_with_support(report, threshold, name)

    cm = confusion_matrix(y_test_flat, y_pred_flat, labels=labels, normalize= 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f"/home/npapadopoulou/wat/{name}/cm/confusion_matrix_th{threshold}.png")
    return cm

# Callable fcn
def get_cm(name,label_list, labels): 
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for t in thresholds: 
        the_cm = cm(name, t, label_list, labels)
        pd.DataFrame(the_cm, index=label_list, columns=label_list).to_csv(f"/home/npapadopoulou/wat/{name}/cm/confusion_matrix_th{t}.csv",float_format='%.2f',index=False, header=False)



# labels = [0, 1, 2]
# label_list =["NONE","Double_Tap","Triple_Tap"]

# get_cm("retake_n32_1_2_C16C8C8_simple3_1rej8_seed21_bs256_initLr001", label_list, labels)

