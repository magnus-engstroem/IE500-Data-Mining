import numpy as np
import pandas as pd
from rebalance_train_set import rebalance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train_path = 'train.csv'
train_df = pd.read_csv(train_path)

y_train = train_df['is_benign']
X_train = train_df.loc[:,~train_df.columns.str.contains('is_benign', case=False)]

X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.25
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)



X_train, y_train = rebalance(X_train, y_train, 15000, 85000)

def test_SVM(C, gamma, X_train, Y_train, X_valid, Y_valid):
    svm = SVC(gamma=gamma, C=C, verbose = True)
    svm.fit(X_train, Y_train)

    y_pred = svm.predict(X_valid)

    # --- Evaluate performance ---
    accuracy = accuracy_score(Y_valid, y_pred)
    precision = precision_score(Y_valid, y_pred)
    recall = recall_score(Y_valid, y_pred)
    f1 = f1_score(Y_valid, y_pred)

    return accuracy, precision, recall, f1




Cs = []
for i in range(-5, 15, 2):
    Cs.append(2**(i))

gammas = []
for j in range(-15, 4, 2):
    gammas.append(2**(j))

accuracies = np.zeros((len(Cs), len(gammas)))
precisions = np.zeros((len(Cs), len(gammas)))
recalls = np.zeros((len(Cs), len(gammas)))
f1s = np.zeros((len(Cs), len(gammas)))

for i, C in enumerate(Cs):
    for j, gamma in enumerate(gammas):
        accuracies[i,j], precisions[i,j], recalls[i,j], f1s[i,j] = test_SVM(C, gamma, X_train, y_train, X_val, y_val)

np.savetxt('accuracies_grid.csv', accuracies)
np.savetxt('precisions_grid.csv', precisions)
np.savetxt('recalls_grid.csv', recalls)
np.savetxt('f1s_grid.csv', f1s)
