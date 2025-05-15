import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data_per_student.csv")  # Change to your actual file name

# Drop unwanted columns before correlation
df_corr = df.drop(columns=['student.id', 'student.isTec21'])

# Compute correlation matrix
corr_matrix = df_corr.corr(numeric_only=True)

# Unstack the matrix to get pairwise correlations
corr_pairs = corr_matrix.unstack()

# Drop self-correlations
corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]

# Sort by absolute correlation value in descending order
sorted_corr = corr_pairs.abs().sort_values(ascending=False)

# Drop duplicate pairs (since correlation matrix is symmetric)
sorted_corr = sorted_corr[~sorted_corr.duplicated()]

# Show top 10 highest correlations (with original values, not just absolute)
top_10 = corr_pairs.loc[sorted_corr.head(10).index]
print(top_10)

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    annot_kws={"size": 7}  # smaller annotation text
)
plt.title('Correlation Matrix', fontsize=8)  # smaller title
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()

'''
target = 'student.isGraduated_max'
if target not in df.columns:
    raise ValueError(f"'{target}' column not found in dataset.")

df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=['category', 'object']).columns:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

df_encoded = df_encoded.dropna(subset=[target])
df_encoded = df_encoded.dropna()

correlations = df_encoded.corr(numeric_only=True)[target].sort_values(ascending=False)

correlations = correlations.drop(target)

print("Top correlations:\n")
print(correlations.head(15))
'''