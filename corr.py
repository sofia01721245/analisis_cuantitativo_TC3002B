import pandas as pd

df = pd.read_csv("cleaned_data_per_student.csv")  # Change to your actual file name

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
