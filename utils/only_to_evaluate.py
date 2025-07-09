import numpy as np
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from utils.ascii import ascii_border
from utils.eval_only import get_cm
from utils.plots import plot_snippets, plot_multitrain
import matplotlib.pyplot as plt
from rich.progress import Progress
from rich import print
from rich.console import Console

label_map = {
    "NONE": 0,
    "Double_Tap": 1,
    "Triple_Tap": 2,
}
num_classes = len(set(label_map.values()))
labels = [0, 1, 2]
label_list = list(label_map.keys())


def cm(name, y_pred_probs, y_test, threshold, label_list, labels):
    
    y_pred = np.argmax(y_pred_probs, axis=-1)  #(N, 1600)
    y_pred_max_probs = np.max(y_pred_probs, axis=-1)    
    y_pred = np.where(y_pred_max_probs >= threshold, y_pred, 0) # Where the model is NOT confident enough, I will put it as "NONE"

    y_pred_flat = y_pred.flatten()  #
    y_test_flat = y_test.flatten()

    cm = confusion_matrix(y_test_flat, y_pred_flat, labels=labels, normalize= 'true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f"/home/npapadopoulou/to_eval/{name}/cm/confusion_matrix_th{threshold}.png")

# Callable fcn
def get_cm(name, y_pred_probs, y_test, label_list, labels): 
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    for t in thresholds: 
        cm(name, y_pred_probs, y_test, t, label_list, labels)

def plot_snippets(X_test, y_test, name, y_pred_probs, y_pred, all = True):
    total = len(y_test) if all else 100
    
    print("\n[bold green]Visualizing sample predictions from all classes.\nProcessing...[/bold green]")

    seen_classes = set()
    sampled_indices = []

    # Flatten labels to check which samples contain each class
    for i in range(len(y_test)):
        unique_classes = set(y_test[i].flatten())
        if not unique_classes.issubset(seen_classes):
            sampled_indices.append(i)
            seen_classes.update(unique_classes)
        if len(sampled_indices) >= total:              #here#
            break

    # Pad up to 50 if not yet enough samples
    if len(sampled_indices) < total:                   #here#
        remaining = list(set(range(total)) - set(sampled_indices))
        sampled_indices += remaining[:(total - len(sampled_indices))]              #here#
    with Progress() as p:
        task = p.add_task("[cyan]Plotting snippets...", total=len(sampled_indices))
        
        # Now plot only those selected samples
        for i in sampled_indices:
            data_sample = X_test[i]
            true_labels = y_test[i]
                
            predicted_labels = y_pred[i]
            pred_probs = y_pred_probs[i, :, :]  # shape (T, num_classes)

            snippet_len = len(true_labels)
            colors = {
                    0: "#3F67A8", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"
                } #colorhexa colours

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [0.2, 0.2, 1.5, 3]})

            for j, val in enumerate(true_labels):
                color = colors.get(val, 'gray')
                ax1.barh(y=0, width=1, left=j, height=0.1, color=color, alpha = 0.6)
            ax1.set_xlim(0, snippet_len)
            ax1.set_yticks([0])
            ax1.set_yticklabels(["True"], fontsize = "10")
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.tick_params(left=False, bottom=False)
                
                # Plot predicted labels
            for j, val in enumerate(predicted_labels):
                color = colors.get(val, 'gray')
                ax2.barh(y=0, width=1, left=j, height=0.1, color=color, alpha = 0.6)
            ax2.set_xlim(0, snippet_len)
            ax2.set_yticks([0])
            ax2.set_yticklabels(["Pred"], fontsize=10)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.tick_params(left=False, bottom=False)

            n_axes = data_sample.shape[1]
            channel_names = ["Axis x", "Axis y", "Axis z", "Jerk x", "Jerk y", "Jerk z"]
            for axis in range(min(n_axes, len(channel_names))):
                ax3.plot(data_sample[:, axis], label=channel_names[axis], color='black', alpha=0.8, linewidth=1)
            # ax3.set_ylim(-4,4)
            ax3.legend(loc = 'upper right')
            ax3.grid(True)
            
            proba = y_pred_probs[i]
            for class_idx in range(3): #3 - number of classes
                ax4.plot(proba[:, class_idx], label=f"Class {class_idx}", alpha=0.9, color = colors[class_idx] , linewidth= 2)
            ax4.plot(true_labels, label="True", linestyle="--", color="black", linewidth= 2, alpha = 1)
            ax4.plot(predicted_labels, label="Predicted",linestyle="--",color="red", linewidth= 2, alpha = 1)
            ax4.set_xlim(0, snippet_len)
            ax4.set_ylim(0.00, 2.00)
            ax4.legend(loc = 'upper right')
            ax4.grid(True)

            plt.tight_layout()
            plt.savefig(f"/home/npapadopoulou/to_eval/{name}/just_plotted/sample_{i}_probs_vs_true.png")
            plt.close("all")

            p.update(task, advance=1)  


def evaluate_and_visualize(X_test, y_test, label_list, labels,  name):
    os.makedirs(f"/home/npapadopoulou/to_eval/{name}/cm", exist_ok=True)
    os.makedirs(f"/home/npapadopoulou/to_eval/{name}/just_plotted", exist_ok=True)
    print(f"\n[bold green]Evaluating model: {name}[/bold green]")
    
    y_pred_probs = np.load(f"/home/npapadopoulou/wat/eval_mltools/{name}/y_pred.npy") #predicted probabilities
    # y_test = np.load(f"/home/npapadopoulou/wat/eval_mltools/{name}/y_test.npy") #ground truth labels

    threshold = 0.7  # Only keeps predictions where the model is 70% confident
    y_pred = np.argmax(y_pred_probs, axis=-1)  #(N, 1600)
    y_pred_max_probs = np.max(y_pred_probs, axis=-1)    
    y_pred = np.where(y_pred_max_probs >= threshold, y_pred, 0) # Where the model is NOT confident enough, I will put it as "NONE"

    y_pred_flat = y_pred.flatten()  #
    y_test_flat = y_test.flatten()

    ascii_border("Classification Report", "double")
    print(classification_report(y_test_flat, y_pred_flat, target_names=label_list, labels=labels, zero_division=0))

    ascii_border("Confusion Matrix", "double")
    cm = confusion_matrix(y_test_flat, y_pred_flat, labels=labels, normalize= 'true')
    print(cm)
    get_cm(name, y_pred_probs, y_test, label_list, labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f"/home/npapadopoulou/to_eval/{name}/just_plotted/confusion_matrix_for_th{threshold}.png")
   
    # plot_snippets(X_test, y_test, name, y_pred_probs, y_pred, all=False)

# -------------------------- To use --------------------------
X_test_path = "/home/npapadopoulou/snippets_folder/simple5_snippets_ped150_rej8/ML_accel/x_test.npy"
y_test_path = "/home/npapadopoulou/snippets_folder/simple5_snippets_ped150_rej8/ML_accel/y_test.npy"
name = "n42_13_oneconv1etvic_simple4rej8_C_bs512_noseed_scaledfortraintest_0.01" 
X_test = np.load(X_test_path)
y_test = np.load(y_test_path)
evaluate_and_visualize(X_test, y_test, label_list, labels,  name)

