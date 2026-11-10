# Food Freshness Classifier

A deep learning image classification project that identifies the type of fruit or vegetable and classifies its **visual freshness condition** as fresh or stale.

The project uses transfer learning with **MobileNetV3 Small** and provides a **Streamlit web interface** where users can upload an image and receive a prediction with confidence scores.

> **Note:** This project performs visual freshness classification. It does not determine whether food is safe to eat.

---

## Features

- Classifies 10 food freshness categories
- Identifies both food type and freshness condition
- Uses transfer learning with MobileNetV3 Small
- Handles class imbalance using weighted Cross-Entropy Loss
- Uses training-time image augmentation
- Provides Top-3 predictions with confidence scores
- Includes a Streamlit web application
- Evaluated on a held-out test set
- Supports external image testing

---

## Classes

The model predicts the following 10 classes:

- Fresh Apple
- Fresh Banana
- Fresh Bitter Gourd
- Fresh Capsicum
- Fresh Orange
- Fresh Tomato
- Stale Apple
- Stale Banana
- Stale Bitter Gourd
- Stale Capsicum

---

## Dataset

The project uses the Kaggle **Fresh and Stale Images of Fruits and Vegetables** dataset.

The dataset contains:

- **11,243 images**
- **10 classes**

The original dataset is imbalanced, with some classes containing significantly fewer images than others.

The data was divided using a stratified split:

- Training: **7,870 images**
- Validation: **1,686 images**
- Test: **1,687 images**

The dataset itself is not included in this repository because of its size.

---

## Machine Learning Pipeline

```text
Raw Images
     ↓
Train / Validation / Test Split
     ↓
Image Resize: 224 × 224
     ↓
Data Augmentation
     ↓
Tensor Conversion
     ↓
ImageNet Normalization
     ↓
MobileNetV3 Small
     ↓
Transfer Learning
     ↓
10-Class Classification
     ↓
Prediction + Confidence
```

---

## Model

The project uses **MobileNetV3 Small** with pretrained ImageNet weights.

The pretrained feature extractor is frozen and the final classifier is replaced with a classifier for the project's 10 classes.

### Why MobileNetV3 Small?

MobileNetV3 Small was selected because it is:

- Lightweight
- Computationally efficient
- Suitable for CPU-based inference
- Faster to train and deploy than larger CNN architectures
- Suitable for a small-scale image classification application

---

## Preprocessing

Input images are resized to:

```text
224 × 224 pixels
```

Training images use augmentation:

- Random horizontal flip
- Random rotation

Validation and test images use deterministic preprocessing.

Images are normalized using ImageNet statistics:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## Handling Class Imbalance

The dataset contains a large difference in the number of images between classes.

For example, the training set contains many more stale apple images than stale capsicum images.

To reduce the effect of this imbalance, class-weighted Cross-Entropy Loss was used.

The class weight for each class was calculated as:

```text
weight = total_samples / (number_of_classes × class_samples)
```

This gives greater importance to classes with fewer training examples.

---

## Training

The model was trained for 5 epochs using:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Loss: Weighted Cross-Entropy Loss
- Hardware: CPU

The best model was selected using validation accuracy.

The trained model is saved locally as:

```text
best_model.pth
```

The model file is excluded from GitHub because of its size.

---

## Results

The final model achieved:

### Test Accuracy

**99.53%**

on the held-out test set of **1,687 images**.

### Additional Metrics

- Macro F1-score: **99.71%**
- Weighted F1-score: **99.52%**

The model made only **8 incorrect predictions out of 1,687 test images**.

The main confusion occurred between fresh and stale apples, which is reasonable because these classes can have visually similar characteristics.

> The reported accuracy is based on the project's held-out test dataset and does not guarantee the same performance on arbitrary real-world images.

---

## Web Application

The project includes a Streamlit application.

The application allows the user to:

1. Upload a food image
2. View the uploaded image
3. Automatically receive a prediction
4. See the predicted food type
5. See the predicted freshness condition
6. See the confidence score
7. View the Top-3 predictions

Example workflow:

```text
Upload Image
      ↓
Model Prediction
      ↓
Food: Capsicum
      ↓
Condition: Fresh
      ↓
Confidence: 99.61%
      ↓
Top 3 Predictions
```

---

## Project Structure

```text
ML project/
│
├── app.py
├── model.py
├── food_dataset.py
├── prepare_dataset.py
├── train.py
├── evaluate.py
├── predict.py
├── plot_confusion_matrix.py
│
├── check_dataset.py
├── class_distribution.py
├── view_samples.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── confusion_matrix.png
│
├── dataset/
│   └── archive/
│       └── ...
│
├── external_test_img/
│   └── ...
│
└── best_model.pth
```

The dataset, virtual environment, and trained model are excluded from version control.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "ML project"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the dataset and place it inside:

```text
dataset/archive/
```

The expected structure is:

```text
dataset/
└── archive/
    ├── fresh_apple/
    ├── fresh_banana/
    ├── fresh_bitter_gourd/
    ├── fresh_capsicum/
    ├── fresh_orange/
    ├── fresh_tomato/
    ├── stale_apple/
    ├── stale_banana/
    ├── stale_bitter_gourd/
    └── stale_capsicum/
```

---

## Preparing the Dataset

Run:

```powershell
python prepare_dataset.py
```

This creates:

```text
dataset_splits.csv
```

containing the image paths, labels, and train/validation/test split assignments.

---

## Training the Model

Run:

```powershell
python train.py
```

The best validation model will be saved as:

```text
best_model.pth
```

---

## Evaluating the Model

Run:

```powershell
python evaluate.py
```

This generates evaluation metrics including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

---

## Single Image Prediction

You can also test an individual image:

```powershell
python predict.py "path\to\image.jpg"
```

Example:

```powershell
python predict.py "external_test_img\banana.jpg"
```

The program displays the predicted class, confidence, and Top-3 predictions.

---

## Running the Streamlit Application

Start the web application with:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

The application automatically performs prediction after an image is uploaded.

---

## Limitations

This project has several limitations:

- The dataset is relatively small for a production-grade computer vision system.
- The dataset is class-imbalanced.
- Some classes contain very few images.
- Real-world images can have different lighting, backgrounds, camera angles, and image quality.
- The model only recognizes the 10 classes present in the training dataset.
- A high model confidence does not mean the prediction is always correct.
- Visual freshness classification should not be interpreted as a food safety assessment.

---

## Future Improvements

Possible improvements include:

- Add more food categories
- Collect a larger and more diverse dataset
- Improve performance on rare classes
- Fine-tune more MobileNet layers
- Compare MobileNetV3 with ResNet or EfficientNet
- Add Grad-CAM visual explanations
- Add confidence-based uncertainty warnings
- Deploy the application online
- Optimize the model for mobile or edge devices

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- MobileNetV3
- Transfer Learning
- Scikit-learn
- Pandas
- NumPy
- Pillow
- Matplotlib
- OpenCV
- Streamlit

---

## Project Outcome

This project demonstrates an end-to-end computer vision workflow:

```text
Dataset
   ↓
Exploratory Data Analysis
   ↓
Data Preparation
   ↓
Preprocessing
   ↓
Transfer Learning
   ↓
Model Training
   ↓
Validation
   ↓
Test Evaluation
   ↓
External Testing
   ↓
Streamlit Deployment Interface
```

It combines machine learning, computer vision, model evaluation, and application development into a single end-to-end project.