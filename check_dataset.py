from pathlib import Path

dataset_path = Path("dataset/archive")

total_images = 0

for folder in sorted(dataset_path.iterdir()):
    if folder.is_dir():
        images = [
            file for file in folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(f"{folder.name}: {len(images)} images")
        total_images += len(images)

print("\nTotal images:", total_images)
