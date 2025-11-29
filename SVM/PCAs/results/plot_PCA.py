import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


no_rebal_data = np.loadtxt("Proj/IE500-Data-Mining/SVM_analysis/PCAs/no_rebalance.csv", delimiter=",")
ratio_10_90_data = np.loadtxt("Proj/IE500-Data-Mining/SVM_analysis/PCAs/ratio_10_90.csv", delimiter=",")


def plot_score(score_array, title, ax):
    num_pcs = score_array[2, :]
    accuracy = score_array[3, :]
    precision = score_array[4, :]
    recall = score_array[5, :]
    f1_scores = score_array[6, :]

    ax.plot(num_pcs, accuracy, marker='o', label='Accuracy')
    ax.plot(num_pcs, precision, marker='s', label='Precision')
    ax.plot(num_pcs, recall, marker='^', label='Recall')
    ax.plot(num_pcs, f1_scores, marker='d', label='F1-score')

    # Formatting
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Number of Principal Components", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_score(no_rebal_data, "No rebalancing", axes[0])
plot_score(ratio_10_90_data, r"10% benign, 90% malicious", axes[1])

plt.tight_layout()
plt.savefig("PCA_subplots.png")
