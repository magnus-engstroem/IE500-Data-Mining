from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.utils import shuffle
import pandas as pd
import numpy as np



def rebalance(X, y, no_1, no_0):
    """
    Clean and rebalance a binary classification dataset using
    Edited Nearest Neighbours (ENN), random undersampling, and SMOTE.

    This function first removes potentially noisy samples from the 
    majority class using ENN, then optionally undersamples the majority 
    class to a target count, and finally oversamples the minority class 
    using SMOTE to achieve a balanced dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix of the input data.

    y : array-like of shape (n_samples,)
        Target vector corresponding to class labels. Must be binary 
        (typically 0 for majority class and 1 for minority class).

    no_1 : int
        Desired number of minority (benign = label 1) samples.

    no_0 : int
        Desired number of majority class (malign = label 0) samples after resampling.
        Undersampling will only occur if the current number of class 0 
        samples exceeds this value.

    Returns
    -------
    X_resampled : ndarray of shape (n_resampled_samples, n_features)
        The resampled feature matrix after cleaning and rebalancing.

    y_resampled : ndarray of shape (n_resampled_samples,)
        The corresponding resampled target vector.

        
    
    Raises
    ------
    ValueError
        If the requested resampling targets would result in
        downsampling the minority class or upsampling the majority class.


    Notes
    -----
    The resampling process consists of three sequential steps:
        1. **Edited Nearest Neighbours (ENN)**: removes noisy samples 
           from the majority class.
        2. **Random undersampling**: reduces the majority class size 
           if it exceeds `no_0`.
        3. **SMOTE**: generates synthetic samples for the minority class 
           until it reaches `no_1`.

    Examples
    --------
    X_bal, y_bal = rebalance(X, y, no_1=500, no_0=500)
    print(np.bincount(y_bal))
    [500 500]
    """


        # Validate binary class labels
    unique_classes = np.unique(y)
    if set(unique_classes) != {0, 1}:
        raise ValueError("y must contain exactly two classes labeled 0 and 1.")

    # Count current class sizes
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)

    # Check for invalid target directions
    if no_0 > n0:
        raise ValueError(
            f"Invalid target: no_0={no_0} would upsample the majority class"
        )
    if no_1 < n1:
        raise ValueError(
            f"Invalid target: no_1={no_1} would downsample the minority class."
        )

    X_down, y_down = EditedNearestNeighbours(sampling_strategy = 'majority').fit_resample(X, y)
    if np.sum(1-y_down) > no_0:
        X_down, y_down = RandomUnderSampler(sampling_strategy = {0:no_0}).fit_resample(X_down, y_down)
    X_upsampled, y_upsampled = SMOTE(sampling_strategy = {1:no_1}).fit_resample(X_down, y_down)
    X_upsampled, y_upsampled = shuffle(X_upsampled, y_upsampled, random_state=42)
    return X_upsampled, y_upsampled



if __name__ == '__main__':
    #Test set
    train_path = 'train.csv'
    train_df = pd.read_csv(train_path)
    y_train = train_df['is_benign']
    X_train = train_df.loc[:,~train_df.columns.str.contains('is_benign', case=False)]


    #Upsampling from 10758 benign samples in train set
    n_benign = 40000

    #Downsampling from 150704 malignant samples in training set
    n_mal = 60000

    X_re, y_re = rebalance(X_train, y_train, 40000, 60000)
