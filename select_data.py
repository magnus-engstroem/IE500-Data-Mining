import pandas as pd
import numpy as np

#Files
feature_file = 'Proj/Data/features2.txt'   
input_csv = 'Proj/Data/Android Malware Analysis/Android Malware Analysis CSV Dataset/CIC-AndMal2017.csv'            # your original data file (can also be loaded as df)
output_csv = 'selected_data.csv'        
target_col = 'Label'                    

# --- Load the feature list ---
with open(feature_file, 'r') as f:
    features_to_keep = [line.strip() for line in f if line.strip()]

# --- Load your data ---
df = pd.read_csv(input_csv)

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
filtered_df.to_csv(output_csv, index=False)

print(f"Saved {len(filtered_df.columns)} columns and {len(filtered_df)} rows to '{output_csv}'")
