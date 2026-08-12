import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

project_folder = Path(__file__).resolve().parent

input_file = project_folder.parent / "UV_vis_merged.csv"

output_folder = project_folder / "data"
output_folder.mkdir(exist_ok=True)

df = pd.read_csv(input_file)

train_df, test_df = train_test_split(
    df,
    test_size=0.10,
    random_state=42,
    shuffle=True
)

train_df, validation_df = train_test_split(
    train_df,
    test_size=0.10,
    random_state=42,
    shuffle=True
)

train_df.to_csv(
    output_folder / "train_transformer.csv",
    index=False
)

validation_df.to_csv(
    output_folder / "validation_transformer.csv",
    index=False
)

test_df.to_csv(
    output_folder / "test_transformer.csv",
    index=False
)

print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))