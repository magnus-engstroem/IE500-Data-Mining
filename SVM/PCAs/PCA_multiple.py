from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from rebalance_train_set import rebalance



train_path = 'train.csv'
train_df = pd.read_csv(train_path)
test_path = 'test.csv'
test_df = pd.read_csv(test_path)


Y_train = train_df['is_benign']
X_train = train_df.loc[:,~train_df.columns.str.contains('is_benign', case=False)]

Y_test = test_df['is_benign']
X_test = test_df.loc[:,~test_df.columns.str.contains('is_benign', case=False)]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




def test_score(X_train, Y_train, X_test, Y_test, rebalance, ben, mal, PCs):

    if rebalance:
        X_train, Y_train = rebalance(X_train, Y_train, ben, mal)

    pca = PCA(n_components=PCs)  # Adjust the number of components as needed
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    svm = SVC(gamma='auto', verbose = True)
    svm.fit(X_train, Y_train)

    y_pred = svm.predict(X_test)

    # --- Evaluate performance ---
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)

    return np.array([ben, mal, PCs, accuracy, precision, recall, f1])

def plot_score(score_array, title):

    num_pcs = score_array[2, :]
    accuracy = score_array[3, :]
    precision = score_array[4, :]
    recall = score_array[5, :]
    f1_scores = score_array[6, :]



    plt.figure(figsize=(8, 5))
    plt.plot(num_pcs, accuracy, marker='o', label='Accuracy')
    plt.plot(num_pcs, precision, marker='s', label='Precision')
    plt.plot(num_pcs, recall, marker='^', label='Recall')
    plt.plot(num_pcs, f1_scores, marker='d', label='F1-score')

    # Formatting
    plt.title(title, fontsize=14)
    plt.xlabel("Number of Principal Components", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(title + "_plot.png")

results = np.zeros((7, 30))

for pcs in range(30):
    print(f"PCs: {pcs}")
    results[:,pcs] = test_score(X_train, Y_train, X_test, Y_test, False, 0, 0, pcs + 1)
np.savetxt('no_rebalance.csv', results, delimiter=',')
plot_score(results, "No rebalancing")

for pcs in range(30):
    print(f"PCs: {pcs}")
    results[:,pcs] = test_score(X_train, Y_train, X_test, Y_test, False, 15000, 135000, pcs + 1)
np.savetxt('ratio_10_90.csv', results, delimiter=',')
plot_score(results, "10/90 ratio")

for pcs in range(30):
    print(f"PCs: {pcs}")
    results[:,pcs] = test_score(X_train, Y_train, X_test, Y_test, False, 35000, 140000, pcs + 1)
np.savetxt('ratio_20_80.csv', results, delimiter=',')
plot_score(results, "20/80 ratio")

for pcs in range(30):
    print(f"PCs: {pcs}")
    results[:,pcs] = test_score(X_train, Y_train, X_test, Y_test, False, 100000, 100000, pcs + 1)
np.savetxt('ratio_50_50.csv', results, delimiter=',')
plot_score(results, "50/50 ratio")