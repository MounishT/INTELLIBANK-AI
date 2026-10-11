from pathlib import Path

import numpy as np
import pandas as pd

# Create synthetic Kaggle-like credit card fraud dataset
np.random.seed(42)

n_samples = 50000  # Smaller but sufficient for training
n_fraud = int(n_samples * 0.0017)  # ~0.17% fraud rate like original

# Generate normal transactions (Class=0)
n_normal = n_samples - n_fraud

# Time feature: seconds elapsed between transactions
time_normal = np.cumsum(np.random.exponential(scale=60, size=n_normal))
time_fraud = np.cumsum(np.random.exponential(scale=60, size=n_fraud))

# Amount: log-normal distribution for normal, higher for fraud
amount_normal = np.random.lognormal(mean=3.0, sigma=1.5, size=n_normal)
amount_fraud = np.random.lognormal(mean=4.5, sigma=1.8, size=n_fraud)

# V1-V28: PCA features - normally distributed for normal, different for fraud
v_normal = np.random.randn(n_normal, 28) * 1.5
v_fraud = np.random.randn(n_fraud, 28) * 2.5 + np.random.randn(28) * 2  # Different distribution

# Create DataFrames
df_normal = pd.DataFrame(v_normal, columns=[f'V{i}' for i in range(1, 29)])
df_normal['Time'] = time_normal
df_normal['Amount'] = amount_normal
df_normal['Class'] = 0

df_fraud = pd.DataFrame(v_fraud, columns=[f'V{i}' for i in range(1, 29)])
df_fraud['Time'] = time_fraud[:n_fraud]
df_fraud['Amount'] = amount_fraud
df_fraud['Class'] = 1

# Combine and shuffle
df = pd.concat([df_normal, df_fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Reorder columns to match original: Time, V1-V28, Amount, Class
cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
df = df[cols]

# Save
output_path = Path(__file__).parent / 'data' / 'raw' / 'creditcard.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f'Created synthetic dataset: {len(df)} rows, {df["Class"].sum()} fraud cases ({df["Class"].mean()*100:.4f}%)')
print('Columns:', df.columns.tolist())
