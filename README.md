# Food Freshness Classifier

A deep learning application that classifies food images based on their **visual freshness condition** using transfer learning with **MobileNetV3 Small**.

## Live Demo

**Try the deployed application:**

https://fruits-health-based-on-picture-ferers8ytrsqpym4qjjnff.streamlit.app/

Upload a food image and the application automatically predicts the food type, freshness condition, and confidence.

> **Note:** This model performs visual freshness classification. It does not determine whether food is safe to eat.

---

## Project Overview

Freshness classification can help automate the visual inspection of fruits and vegetables. This project uses a convolutional neural network with transfer learning to classify images into food-specific fresh and stale categories.

The complete pipeline covers:

**Dataset → Data Preparation → Preprocessing → Data Augmentation → Transfer Learning → Training → Evaluation → Prediction → Streamlit Deployment**

---

## Features

- Classifies food images into fresh/stale categories
- Supports 10 food freshness classes
- Uses MobileNetV3 Small with transfer learning
- Handles class imbalance using weighted Cross-Entropy Loss
- Uses training data augmentation
- Provides confidence scores for predictions
- Includes a trained model
- Includes test-set evaluation and confusion matrix
- Provides a Streamlit web application
- Automatically predicts after an image is uploaded
- Deployed publicly using Streamlit Community Cloud

---

## Supported Classes

The model was trained on the following 10 classes:

```text
fresh_apple
fresh_banana
fresh_bitter_gourd
fresh_capsicum
fresh_orange
fresh_tomato
stale_apple
stale_banana
stale_bitter_gourd
stale_capsicum
```

---

## Dataset

The project uses the **Fresh and Stale Images of Fruits and Vegetables** dataset.

### Dataset Statistics

- Total images: **11,243**
- Number of classes: **10**

The dataset was divided into:

| Split | Images |
|---|---:|
| Training | 7,870 |
| Validation | 1,686 |
| Test | 1,687 |

The dataset contains significant class imbalance. For example, the training set contains many more stale apples than stale capsicums.

To address this, class weights were calculated from the training distribution and used with Cross-Entropy Loss.

---

## Machine Learning Pipeline

```text
Raw Images
     ↓
Dataset Analysis
     ↓
Train / Validation / Test Split
     ↓
Image Preprocessing
     ↓
Data Augmentation
     ↓
MobileNetV3 Small
     ↓
Transfer Learning
     ↓
Weighted Cross-Entropy Loss
     ↓
Training
     ↓
Validation
     ↓
Best Model Selection
     ↓
Test Evaluation
     ↓
Streamlit Deployment
```

---

## Image Preprocessing

All images are:

1. Resized to **224 × 224**
2. Converted to RGB
3. Converted to PyTorch tensors
4. Normalized using ImageNet statistics

### Training Augmentation

Training images use:

- Random horizontal flipping
- Random rotation up to 10 degrees

Validation and test images use only deterministic preprocessing.

This prevents artificial transformations from affecting evaluation results.

---

## Model

The project uses **MobileNetV3 Small** pretrained on ImageNet.

Instead of training the entire neural network from scratch, transfer learning is used.

### Transfer Learning Approach

The pretrained feature extractor is frozen:

```text
MobileNetV3 Small
      ↓
Pretrained Feature Extractor
      ↓
Classifier
      ↓
10 Output Classes
```

The final classifier is modified to produce predictions for the project's 10 classes.

### Why MobileNetV3 Small?

MobileNetV3 Small was selected because it provides a good balance between:

- Model size
- Computational efficiency
- Prediction speed
- Accuracy
- CPU compatibility
- Deployment suitability

This makes it practical for a lightweight portfolio application.

---

## Handling Class Imbalance

The original dataset contains highly uneven class distributions.

For example:

```text
stale_apple          2342 images
stale_capsicum         39 images
```

Training directly on this distribution could cause the model to favor classes with more examples.

To reduce this effect, class weights were calculated using:

```text
weight_i = N / (C × n_i)
```

where:

- `N` = total number of training samples
- `C` = number of classes
- `n_i` = number of samples belonging to class i

These weights were used with weighted Cross-Entropy Loss.

---

## Training

### Training Configuration

```text
Model:              MobileNetV3 Small
Transfer Learning:  Yes
Epochs:             5
Batch Size:         32
Optimizer:          Adam
Learning Rate:      0.001
Loss:               Weighted Cross-Entropy
Device:             CPU
Input Size:         224 × 224
```

The model with the best validation accuracy was saved as:

```text
best_model.pth
```

---

## Results

The final trained model was evaluated on a **held-out test set of 1,687 images**.

### Test Performance

| Metric | Score |
|---|---:|
| Accuracy | **99.53%** |
| Macro F1 Score | **99.71%** |
| Weighted F1 Score | **99.52%** |

The model made **8 incorrect predictions out of 1,687 test images**.

### Main Confusions

Most of the incorrect predictions involved visually similar apple classes:

```text
fresh_apple → stale_apple
stale_apple → fresh_apple
```

This is expected because freshness can sometimes be difficult to distinguish from appearance alone.

The `stale_capsicum` test class achieved 6/6 correct predictions, but this class has very few examples, so this result should not be interpreted as evidence of broad real-world performance.

---

## Confusion Matrix

The confusion matrix was generated using the held-out test set to analyze class-level performance.

It helps identify which classes the model confuses with one another instead of relying only on overall accuracy.

---

## Streamlit Application

The trained model is integrated into a Streamlit web application.

### Application Flow

```text
Upload Image
     ↓
Resize to 224 × 224
     ↓
Normalize Image
     ↓
MobileNetV3 Small
     ↓
Prediction
     ↓
Food Type + Freshness + Confidence
```

The application automatically performs prediction after an image is uploaded.

No separate classification button is required.

### Example

An uploaded image can produce a result such as:

```text
Food: Apple
Condition: Fresh
Confidence: 98.42%
```

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

https://fruits-health-based-on-picture-ferers8ytrsqpym4qjjnff.streamlit.app/

The deployment uses the trained model stored in the GitHub repository.

---

## Project Structure

```text
Fruits-health-based-on-picture/
│
├── app.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── food_dataset.py
├── prepare_dataset.py
├── check_dataset.py
├── class_distribution.py
├── plot_confusion_matrix.py
│
├── dataset_splits.csv
├── best_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

The original dataset and Python virtual environment are excluded from GitHub using `.gitignore`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NITIKA39/Fruits-health-based-on-picture.git
```

```bash
cd Fruits-health-based-on-picture
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

The dataset is not included in the GitHub repository because of its size.

The expected local structure is:

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

After placing the dataset in the expected location, run:

```bash
python prepare_dataset.py
```

This creates:

```text
dataset_splits.csv
```

containing the image paths, labels, and dataset split information.

---

## Running the Training Pipeline

### Prepare the dataset

```bash
python prepare_dataset.py
```

### Train the model

```bash
python train.py
```

The best validation model is saved as:

```text
best_model.pth
```

### Evaluate the model

```bash
python evaluate.py
```

### Generate the confusion matrix

```bash
python plot_confusion_matrix.py
```

### Test a single image

```bash
python predict.py
```

### Run the Streamlit application locally

```bash
streamlit run app.py
```

The local application will normally be available at:

```text
http://localhost:8501
```

---

## Technologies Used

### Programming

- Python

### Machine Learning

- PyTorch
- Torchvision
- MobileNetV3 Small
- Transfer Learning
- Scikit-learn

### Image Processing

- Pillow
- OpenCV

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib

### Deployment

- Streamlit
- Streamlit Community Cloud

### Version Control

- Git
- GitHub

---

## Limitations

This project has several limitations.

### 1. Dataset limitations

The model was trained on a specific dataset and may not perform equally well on images from completely different environments.

Differences in:

- Lighting
- Background
- Camera quality
- Food orientation
- Image quality
- Food variety

can affect predictions.

### 2. Class coverage

The model only recognizes the 10 classes included in the training dataset.

It should not be assumed to correctly classify foods outside these categories.

### 3. Class imbalance

Some classes contain substantially fewer images than others, especially `stale_capsicum`.

Class weighting was used to reduce the effect of imbalance, but additional data would improve reliability.

### 4. Visual freshness is not food safety

The model only analyzes visual patterns.

A prediction of `fresh` does **not** mean that food is safe to consume.

---

## Future Improvements

Possible improvements include:

- Collecting a larger and more diverse dataset
- Adding more fruits and vegetables
- Increasing the number of examples for rare classes
- Fine-tuning more MobileNet layers
- Comparing MobileNetV3 with EfficientNet or ResNet
- Adding image-quality checks
- Testing on a larger external dataset
- Adding prediction history
- Adding an API backend
- Optimizing the model for mobile deployment
- Adding confidence-based "uncertain" predictions

---

## What I Learned

This project provided hands-on experience with the complete machine learning workflow:

- Dataset exploration
- Class distribution analysis
- Train/validation/test splitting
- Data augmentation
- Image preprocessing
- PyTorch Dataset and DataLoader
- Transfer learning
- CNN classification
- Handling class imbalance
- Model training
- Validation
- Test-set evaluation
- Confusion matrix analysis
- Single-image inference
- Streamlit application development
- Git/GitHub workflow
- Model deployment

---

## Project Outcome

The project demonstrates an end-to-end machine learning application rather than only a trained model.

It covers the complete journey:

```text
Dataset
   ↓
Exploration
   ↓
Preprocessing
   ↓
Model Development
   ↓
Training
   ↓
Evaluation
   ↓
Inference
   ↓
Web Application
   ↓
Cloud Deployment
```

**Final result:** a deployed food freshness classification application powered by a MobileNetV3 Small transfer-learning model.