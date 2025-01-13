import numpy as np
import pandas as pd

# Example arrays
array1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Larger array
array2 = np.array([1, 3, 5, 7])  # Smaller array

# Convert to pandas DataFrame for easier manipulation
df1 = pd.DataFrame(array1, columns=["Data1"])
df2 = pd.DataFrame(array2, columns=["Data2"])

# Step 1: Truncate the larger array if exact matching is required
min_length = min(len(df1), len(df2))
df1_truncated = df1.iloc[:min_length]
df2_truncated = df2.iloc[:min_length]

print("Truncated Matching Arrays:")
print(df1_truncated)
print(df2_truncated)

# Step 2: Interpolating smaller data to match larger array if continuity matters
df2_interpolated = pd.DataFrame(
    np.interp(
        np.linspace(0, len(df2) - 1, len(df1)),  # Target positions in df1 size
        np.arange(len(df2)),  # Original positions in df2 size
        df2["Data2"]  # Data values from df2
    ),
    columns=["Data2"] 
)

# Resetting the index for consistency
df1.reset_index(drop=True, inplace=True)
df2_interpolated.reset_index(drop=True, inplace=True)

print("\nInterpolated Matching Arrays:")
print(df1)
print(df2_interpolated)