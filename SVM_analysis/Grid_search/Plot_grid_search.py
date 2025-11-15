import numpy as np
import matplotlib.pyplot as plt

# --- Load your four CSV files ---
data1 = np.loadtxt("Proj/IE500-Data-Mining/SVM_analysis/Grid_search/accuracies_grid.csv", delimiter=" ")
data2 = np.loadtxt("Proj/IE500-Data-Mining/SVM_analysis/Grid_search/precisions_grid.csv", delimiter=" ")
data3 = np.loadtxt("Proj/IE500-Data-Mining/SVM_analysis/Grid_search/recalls_grid.csv", delimiter=" ")
data4 = np.loadtxt("Proj/IE500-Data-Mining/SVM_analysis/Grid_search/f1s_grid.csv", delimiter=" ")

C_powers = []
for i in range(15, -5, -2):
    C_powers.append(i)

gamma_powers = []
for j in range(-15, 4, 2):
    gamma_powers.append(j)

# Convert exponents to 2^p
x_labels = ["$2^{" + str(p) + "}$" for p in gamma_powers]
y_labels = ["$2^{" + str(p) + "}$" for p in C_powers]

def plot_heatmap(ax, data, title, cmap):
    im = ax.imshow(data, origin='lower', aspect='auto', cmap = cmap)
    ax.set_title(title)
    ax.set_xlabel("γ")
    ax.set_ylabel("C")

    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_yticklabels(y_labels, rotation=45)

    # One colorbar per subplot
    plt.colorbar(im, ax=ax)
    return im

fig, axs = plt.subplots(2, 2, figsize=(8, 7))
fig.suptitle('SVM hyperparameters grid search')


cmaps = ["Blues", "OrRd", "Greens", "Purples"]

plot_heatmap(axs[0,0], data1, "Accuracy", cmaps[0])
plot_heatmap(axs[0,1], data2, "Precision", cmaps[1])
plot_heatmap(axs[1,0], data3, "Recall", cmaps[2])
plot_heatmap(axs[1,1], data4, "F1 score", cmaps[3])

plt.tight_layout()
plt.savefig("grids.png")