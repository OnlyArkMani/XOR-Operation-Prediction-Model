import pandas as pd
import numpy as np

# === Load raw dataset (no header!) ===
file_path = r"C:\Projects\Assignment\data-100000-100-4-rnd.csv"
df = pd.read_csv(file_path, header=None)  # important: header=None

# Split into features (X) and target (y)
X = df.iloc[:, :-1].values.astype(np.uint8)
y = df.iloc[:, -1].values.astype(np.uint8)

n, d = X.shape
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# === Build augmented matrix [X | y] for GF(2) elimination ===
A = np.concatenate([X, y[:, None]], axis=1).astype(np.uint8)

# === Forward elimination over GF(2) ===
rank = 0
pivot_cols = []

for col in range(d):
    # Find pivot row with 1 in this column
    pivot_row = None
    for r in range(rank, n):
        if A[r, col] == 1:
            pivot_row = r
            break
    if pivot_row is None:
        continue

    # Swap pivot row up
    if pivot_row != rank:
        A[[rank, pivot_row]] = A[[pivot_row, rank]]

    pivot_cols.append(col)

    # Eliminate 1's below pivot
    for r in range(rank + 1, n):
        if A[r, col] == 1:
            A[r, :] ^= A[rank, :]

    rank += 1

# === Back substitution to recover solution vector ===
solution = np.zeros(d, dtype=np.uint8)
for i in reversed(range(len(pivot_cols))):
    col = pivot_cols[i]
    row = i
    val = A[row, -1]
    for j in range(col + 1, d):
        if A[row, j] == 1:
            val ^= solution[j]
    solution[col] = val

# === Evaluate recovered XOR ===
y_pred = (X @ solution) % 2
acc = (y_pred == y).mean()
active_cols = np.where(solution == 1)[0]

print(f"\nRecovered XOR columns: {active_cols}")
print(f"Number of active columns: {len(active_cols)}")
print(f"Accuracy of recovered linear combination: {acc * 100:.2f}%")
