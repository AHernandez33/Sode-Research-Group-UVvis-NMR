# view experimental data clean CSV
import pandas as pd
df = pd.read_csv("nmrshiftdb_1H_1000.csv")
print(df.head())