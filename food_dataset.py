import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------------------------
# 1. Read the CSV
# --------------------------------------------------

df = pd.read_csv("dataset_splits.csv")


# --------------------------------------------------
# 2. Separate train, validation and test data
# --------------------------------------------------

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()


# --------------------------------------------------
# 3. Create class mapping
# --------------------------------------------------

classes = sorted(df["label"].unique())

class_to_idx = {
    class_name: index
    for index, class_name in enumerate(classes)
}


print("Classes:")
print(classes)

print("\nClass mapping:")
print(class_to_idx)

print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# --------------------------------------------------
# 4. Image transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 5. Create our custom Dataset
# --------------------------------------------------

class FoodDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["image_path"]
        label_name = row["label"]

        image = Image.open(image_path).convert("RGB")

        label = class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------------------------------
# 6. Create datasets
# --------------------------------------------------

train_dataset = FoodDataset(
    train_df,
    transform=train_transform
)

val_dataset = FoodDataset(
    val_df,
    transform=val_test_transform
)

test_dataset = FoodDataset(
    test_df,
    transform=val_test_transform
)


# --------------------------------------------------
# 7. Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 8. Test one image
# --------------------------------------------------

print("\nNumber of training images:", len(train_dataset))

image, label = train_dataset[0]

print("Image type:", type(image))
print("Image shape:", image.shape)
print("Label:", label)


# --------------------------------------------------
# 9. Test one batch
# --------------------------------------------------

images, labels = next(iter(train_loader))

print("\nBatch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("First 10 labels:", labels[:10])