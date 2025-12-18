import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# === Configuration ===
input_folder = r"C:\Projects\Assignment"
output_folder = os.path.join(input_folder, "preprocessed")
os.makedirs(output_folder, exist_ok=True)

files = [
    "data-100000-100-4-rnd.csv",
    "data-100000-1000-4-rnd.csv",
    "data-100000-10000-4-rnd.csv"
]

CHUNK_SIZE = 10000  # number of rows per chunk, tune as needed

def preprocess_csv_in_chunks(file_path):
    print(f"\n🚀 Preprocessing: {file_path}")
    basename = os.path.basename(file_path)
    output_file = os.path.join(output_folder, basename.replace(".csv", "_preprocessed.csv"))

    # --- First pass: fit scaler incrementally ---
    scaler = StandardScaler()
    total_rows = 0

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        X_chunk = chunk.iloc[:, :-1]
        scaler.partial_fit(X_chunk)
        total_rows += len(chunk)
    print(f"✅ Scaler fitted on {total_rows} rows.")

    # --- Second pass: transform & save ---
    header_written = False
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        X_chunk = chunk.iloc[:, :-1]
        y_chunk = chunk.iloc[:, -1]

        X_scaled = scaler.transform(X_chunk)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_chunk.columns)
        processed_chunk = pd.concat([X_scaled_df, y_chunk.reset_index(drop=True)], axis=1)

        processed_chunk.to_csv(output_file, mode='a', header=not header_written, index=False)
        header_written = True

    print(f"📁 Saved preprocessed file: {output_file}\n")

# Run preprocessing for all files
for f in files:
    preprocess_csv_in_chunks(os.path.join(input_folder, f))
