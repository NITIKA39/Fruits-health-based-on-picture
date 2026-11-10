import torch
import streamlit as st

from PIL import Image
from torchvision import transforms

from model import create_model


# -----------------------------
# Classes
# -----------------------------

CLASSES = [
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


# -----------------------------
# Device
# -----------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Load trained model
# -----------------------------

@st.cache_resource
def load_model():

    model = create_model()

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


model = load_model()


# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Streamlit page
# -----------------------------

st.set_page_config(
    page_title="Food Freshness Classifier",
    page_icon="🍎",
    layout="centered"
)

st.title("Food Freshness Classifier")

st.write(
    "Upload an image to classify the food type "
    "and its visual freshness condition."
)

st.info(
    "This model performs visual freshness classification. "
    "It does not determine whether food is safe to eat."
)


# -----------------------------
# Upload image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

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

    predicted_class = CLASSES[
        predicted_index
    ]

    # Split "stale_banana" into:
    # condition = stale
    # food = banana

    condition, food = predicted_class.split(
        "_",
        maxsplit=1
    )

    # -------------------------
    # Display result
    # -------------------------

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Food",
            food.replace("_", " ").title()
        )

    with col2:

        st.metric(
            "Condition",
            condition.title()
        )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    # -------------------------
    # Top 3 predictions
    # -------------------------

    st.subheader("Top 3 Predictions")

    top_probabilities, top_indices = torch.topk(
        probabilities[0],
        k=3
    )

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        class_name = CLASSES[
            index.item()
        ]

        st.write(
            f"**{class_name.replace('_', ' ').title()}** — "
            f"{probability.item() * 100:.2f}%"
        )