import streamlit as st
import torch

from PIL import Image
from torchvision import transforms

from model import create_model


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Food Freshness Classifier",
    page_icon="🍎",
    layout="centered"
)


# --------------------------------------------------
# Classes
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
# Helper function
# --------------------------------------------------

def format_class_name(class_name):

    parts = class_name.split("_")

    condition = parts[0].capitalize()
    food = " ".join(parts[1:]).title()

    return food, condition


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model()

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model, device


model, device = load_model()


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
# Title
# --------------------------------------------------

st.title("Food Freshness Classifier")

st.write(
    "Upload an image to classify the food type "
    "and its visual freshness condition."
)

st.info(
    "This model performs visual freshness classification. "
    "It does not determine whether food is safe to eat."
)


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=700
    )

    # Convert image to tensor
    input_tensor = transform(image)

    # Add batch dimension
    input_tensor = input_tensor.unsqueeze(0)

    # Move to CPU/GPU
    input_tensor = input_tensor.to(device)


    # ----------------------------------------------
    # Model prediction
    # ----------------------------------------------

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities[0],
            k=3
        )


    # ----------------------------------------------
    # Main prediction
    # ----------------------------------------------

    predicted_index = top_indices[0].item()

    predicted_class = classes[
        predicted_index
    ]

    confidence = top_probabilities[0].item()

    food, condition = format_class_name(
        predicted_class
    )


    # ----------------------------------------------
    # Result
    # ----------------------------------------------

    st.divider()

    st.subheader("Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.caption("Food")

        st.markdown(
            f"### {food}"
        )

    with col2:

        st.caption("Condition")

        if condition == "Fresh":

            st.success(
                f"### {condition}"
            )

        else:

            st.warning(
                f"### {condition}"
            )


    st.caption("Confidence")

    st.markdown(
        f"## {confidence * 100:.2f}%"
    )


    # ----------------------------------------------
    # Top 3 predictions
    # ----------------------------------------------

    # st.divider()

    # st.subheader("Top 3 Predictions")

    # for probability, index in zip(
    #     top_probabilities,
    #     top_indices
    # ):

    #     class_name = classes[
    #         index.item()
    #     ]

    #     food_name, condition_name = format_class_name(
    #         class_name
    #     )

    #     percentage = probability.item() * 100

    #     st.write(
    #         f"**{condition_name} {food_name}**"
    #     )

    #     st.progress(
    #         probability.item()
    #     )

    #     st.caption(
    #         f"{percentage:.2f}%"
    #     )


    # ----------------------------------------------
    # About the model
    # ----------------------------------------------

    # st.divider()

    # with st.expander("About this model"):

    #     st.write(
    #         "This application uses MobileNetV3 Small "
    #         "with transfer learning."
    #     )

    #     st.write(
    #         "The model was trained to classify "
    #         "10 visual food freshness classes."
    #     )

    #     st.write(
    #         "Input images are resized to 224 × 224 pixels "
    #         "and normalized using ImageNet statistics."
    #     )

    #     st.write(
    #         "The model achieved 99.53% accuracy on the "
    #         "held-out test set of 1,687 images."
    #     )

    #     st.write(
    #         "The test accuracy represents performance on "
    #         "the project's held-out dataset and does not "
    #         "guarantee the same performance on arbitrary "
    #         "real-world images."
    #     )