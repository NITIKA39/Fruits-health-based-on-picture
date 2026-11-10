import sys

import torch
from PIL import Image
from torchvision import transforms

from model import create_model


# --------------------------------------------------
# Class names
# --------------------------------------------------

classes = [
    "fresh_apple",
    "fresh_banana",
    "fresh_bitter_gourd",
    "fresh_capsicum",
    "fresh_orange",
    "fresh_tomato",
    "stale_apple",
    "stale_banana",
    "stale_bitter_gourd",
    "stale_capsicum"
]


# --------------------------------------------------
# Check image path
# --------------------------------------------------

if len(sys.argv) != 2:

    print("Usage:")
    print("python predict.py path_to_image")

    sys.exit()


image_path = sys.argv[1]


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = create_model()


# --------------------------------------------------
# Load trained weights
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
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = Image.open(
    image_path
).convert("RGB")


input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_index = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_index
    ].item()


# --------------------------------------------------
# Result
# --------------------------------------------------

predicted_class = classes[
    predicted_index
]

print("\n" + "=" * 50)
print("PREDICTION")
print("=" * 50)

print(
    "Predicted class:",
    predicted_class
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)


# --------------------------------------------------
# Top 3 predictions
# --------------------------------------------------

top_probabilities, top_indices = torch.topk(
    probabilities[0],
    k=3
)

print("\nTop 3 predictions:")

for probability, index in zip(
    top_probabilities,
    top_indices
):

    print(
        f"{classes[index.item()]}: "
        f"{probability.item() * 100:.2f}%"
    )