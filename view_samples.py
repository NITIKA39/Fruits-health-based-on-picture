from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = Path("dataset/archive")

folders = sorted([
    folder for folder in dataset_path.iterdir()
    if folder.is_dir()
])

plt.figure(figsize=(15, 10))

for i, folder in enumerate(folders):
    images = [
        file for file in folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not images:
        continue

    image = Image.open(images[0])

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(folder.name)
    plt.axis("off")

plt.tight_layout()
plt.show()