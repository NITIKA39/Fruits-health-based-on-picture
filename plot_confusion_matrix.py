import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from food_dataset import test_dataset
from model import model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE = 32


# --------------------------------------------------
# Test DataLoader
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
# Load best model
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# Class names
# --------------------------------------------------

df = pd.read_csv("dataset_splits.csv")

classes = sorted(
    df["label"].unique()
)


# --------------------------------------------------
# Get predictions
# --------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# --------------------------------------------------
# Create confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# Plot confusion matrix
# --------------------------------------------------

plt.figure(figsize=(12, 10))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    range(len(classes)),
    classes,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(classes)),
    classes
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title("Food Freshness Classification - Confusion Matrix")


# --------------------------------------------------
# Add numbers inside cells
# --------------------------------------------------

for i in range(len(classes)):

    for j in range(len(classes)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved as:")
print("confusion_matrix.png")