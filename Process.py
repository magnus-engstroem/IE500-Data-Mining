import os
import pandas as pd

#Files
folder_path = "Proj/Data/AndMal2020-dynamic-BeforeAndAfterReboot"      # Folder containing CSV files
features_file = "Proj/Data/features.txt"       # Text file with columns to keep
output_file = "Proj/Data/dynamic_features.csv"  # Output CSV file
chunk_size = 1000      # Number of rows at once

#Select features
with open(features_file, "r") as f:
    features = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(features)} features from {features_file}")

#look throug csv files
csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
print(f"Found {len(csv_files)} CSV files.")

#Only once:
header_written = False

#Process in chunks
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    print(f"\nProcessing {file}...")

    before = 0

    try:
        # Iterate over chunks to avoid loading entire file
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Keep only the columns that exist in both the file and desired features
            available_features = [col for col in features if col in chunk.columns]
            if not available_features:
                continue

            chunk_filtered = chunk[available_features]

            Type = file.split("_")[0]

            if "before" in file:
                before = 1

            chunk_filtered["type"] = Type
            chunk_filtered["before"] = before

            # Write to output file incrementally
            chunk_filtered.to_csv(
                output_file,
                mode="a",  # append
                index=False,
                header=not header_written  # write header only once
            )
            header_written = True

    except Exception as e:
        print(f"Error reading {file}: {e}")

print(f"\nFinished combining CSVs into: {output_file}")
