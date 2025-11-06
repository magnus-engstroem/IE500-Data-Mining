import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Files
feature_file = 'Proj/Data/features2.txt'   
input_csv = 'Proj/Data/Android Malware Analysis/Android Malware Analysis CSV Dataset/CIC-AndMal2017.csv'          
target_col = 'Label'                    

# --- Load the feature list ---
with open(feature_file, 'r') as f:
    features_to_keep = [line.strip() for line in f if line.strip()]

# --- Load your data ---
df = pd.read_csv(input_csv)

#Remove NaN
df = df.dropna()

#Remove duplicates
df = df.drop_duplicates()

#Find and remove faulty rows
benign_label = df['Label'] == 'BENIGN'
not_benign_fam = df['category'] != 'Benign'
bad_label_rows = benign_label & not_benign_fam
df = df[~bad_label_rows]



# --- Filter columns ---
# Ensure features exist in the DataFrame
valid_features = [col for col in features_to_keep if col in df.columns]

# Optionally include the target column
if target_col in df.columns and target_col not in valid_features:
    valid_features.append(target_col)

# Subset the DataFrame
filtered_df = df[valid_features]


#Set the label to a boolean instead of name
filtered_df['Label'] = filtered_df['Label'].map(lambda x: 1 if x == 'BENIGN' else 0)
filtered_df = filtered_df.rename(columns={'Label': 'is_benign'})

# --- Save the result ---
filtered_df.to_csv('selected_data.csv', index=False)

print(f"Saved {len(filtered_df.columns)} columns and {len(filtered_df)} rows to 'selected_data.csv'")

#Split features and labels

X = filtered_df.drop(columns=['is_benign'])
y = filtered_df['is_benign']

# Split into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and labels back into DataFrames
train_df = X_train.copy()
train_df['is_benign'] = y_train

test_df = X_test.copy()
test_df['is_benign'] = y_test

# Save to CSV
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
