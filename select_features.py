import pandas as pd

# --- User settings ---
feature_file = 'Proj/Data/features.txt'   # your text file with one feature name per line
input_csv = 'Proj/Data/Android Malware Analysis/Android Malware Analysis CSV Dataset/CIC-AndMal2017.csv'            # your original data file (can also be loaded as df)
output_csv = 'selected_data.csv'        # output file name
target_col = 'Label'                    # include this in the final dataset

# --- Load the feature list ---
with open(feature_file, 'r') as f:
    features_to_keep = [line.strip() for line in f if line.strip()]

# --- Load your data ---
df = pd.read_csv(input_csv)

# --- Filter columns ---
# Ensure features exist in the DataFrame
valid_features = [col for col in features_to_keep if col in df.columns]

# Optionally include the target column
if target_col in df.columns and target_col not in valid_features:
    valid_features.append(target_col)

# Subset the DataFrame
filtered_df = df[valid_features]

# --- Save the result ---
filtered_df.to_csv(output_csv, index=False)

print(f"Saved {len(filtered_df.columns)} columns and {len(filtered_df)} rows to '{output_csv}'")
