from pathlib import Path
import matplotlib.pyplot as plt

dataset_path = Path("dataset/archive")

class_names = []
image_counts = []

for folder in sorted(dataset_path.iterdir()):
    if folder.is_dir():
        images = [
            file for file in folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        class_names.append(folder.name)
        image_counts.append(len(images))

plt.figure(figsize=(12, 6))

plt.bar(class_names, image_counts)

plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Food Freshness Dataset - Class Distribution")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()