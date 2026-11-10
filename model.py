import torch
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


NUM_CLASSES = 10


def create_model():

    # Load pretrained MobileNetV3 Small
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(
        weights=weights
    )

    # Freeze pretrained feature extractor
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace final classifier
    model.classifier[3] = torch.nn.Linear(
        in_features=1024,
        out_features=NUM_CLASSES
    )

    return model