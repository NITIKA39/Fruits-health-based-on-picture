import torch
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from food_dataset import test_dataset
from model import model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 32


# --------------------------------------------------
# Create Test DataLoader
# --------------------------------------------------

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Load Best Model
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model = model.to(device)


# --------------------------------------------------
# Class Names
# --------------------------------------------------

df = pd.read_csv("dataset_splits.csv")

classes = sorted(
    df["label"].unique()
)

print("\nClasses:")
for index, class_name in enumerate(classes):
    print(f"{index}: {class_name}")


# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# --------------------------------------------------
# Test Accuracy
# --------------------------------------------------

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 50)
print("TEST ACCURACY")
print("=" * 50)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    digits=4,
    zero_division=0
)

print(report)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n" + "=" * 50)
print("CONFUSION MATRIX")
print("=" * 50)

print(cm)


# --------------------------------------------------
# Readable Confusion Matrix
# --------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

print("\nReadable Confusion Matrix:")
print(cm_df)