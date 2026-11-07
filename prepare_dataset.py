from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Location of our dataset
dataset_path = Path("dataset/archive")


# 2. Find all images and their class labels
data = []

for folder in sorted(dataset_path.iterdir()):
    if not folder.is_dir():
        continue

    for file in folder.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            data.append({
                "image_path": str(file),
                "label": folder.name
            })


# 3. Convert our list into a DataFrame
df = pd.DataFrame(data)

print(f"Total images: {len(df)}")
print(f"Total classes: {df['label'].nunique()}")


# 4. Split into training and temporary sets
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)


# 5. Split temporary set into validation and test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)


# 6. Add the split name
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"


# 7. Combine everything
final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# 8. Save the dataset information
final_df.to_csv("dataset_splits.csv", index=False)


# 9. Print split sizes
print("\nSplit sizes:")
print(final_df["split"].value_counts())


# 10. Print class distribution for each split
print("\nClass distribution:")
print(
    final_df.groupby(["split", "label"]).size()
)