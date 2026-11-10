import torch
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from food_dataset import (
    train_dataset,
    val_dataset
)

from model import (
    model,
    criterion,
    optimizer
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_EPOCHS = 5


# --------------------------------------------------
# Create DataLoaders
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


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

model = model.to(device)


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Store loss
        total_loss += loss.item()

        # Get predicted class
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    average_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return average_loss, accuracy


# --------------------------------------------------
# Validation function
# --------------------------------------------------

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Get predicted class
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return average_loss, accuracy


# --------------------------------------------------
# Training loop
# --------------------------------------------------

best_val_accuracy = 0.0

print("\nStarting training...\n")

for epoch in range(NUM_EPOCHS):

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )

    print("-" * 40)

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(
            "Best model saved!"
        )

    print()


print("Training complete.")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)