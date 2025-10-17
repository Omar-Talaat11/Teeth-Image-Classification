# 🦷 Teeth Disease Classification Using Deep Learning

This project is a deep learning pipeline for classifying different teeth conditions and diseases from image data using Convolutional Neural Networks (CNNs) and Transfer Learning (ResNet50).
It also includes a Streamlit web application for real-time image classification.

## 🧠 Project Workflow
1. Data Loading

- Images are loaded using TensorFlow’s image_dataset_from_directory() with:

- Image size: 224×224

- Batch size: 32

2. Data Analysis

- Class distributions are visualized using Matplotlib to understand dataset balance.

3. Data Augmentation

- To enhance model generalization, random augmentations are applied:

    - Horizontal flip

    - Random rotation (±10%)

    - Zoom and contrast variations

4. Balancing Classes

- Data augmentation is repeated to generate balanced datasets.

- Each class is trimmed to a target of 600 samples.

5. Model Architectures

- Model 1 — Basic CNN

    - Conv2D + MaxPooling2D layers

    - Dense layers for classification

    - Softmax output layer with 7 classes

- Model 2 — CNN + GlobalAveragePooling

    - Uses GlobalAveragePooling2D() for dimensionality reduction

    - Simplifies dense layer parameters

- Model 3 — CNN with Residual Blocks

    - Adds residual connections to prevent vanishing gradients

    - Uses Adam(learning_rate=0.0005) optimizer

    - Achieved strong accuracy and stability

- Model 4 — Transfer Learning (ResNet50)

    - Uses pretrained ResNet50 (ImageNet weights) as a feature extractor

    - Adds custom dense layers for fine-tuning teeth disease classification

    - Best-performing model overall

## 📊 Evaluation

- Metrics: Accuracy, Loss, Classification Report, Confusion Matrix

- Dataset splits:

    - Training: Balanced synthetic dataset

    - Validation: Validation/

    - Testing: Testing/

## 🚀 Streamlit Web App

A Streamlit-based UI allows users to upload an image and classify it instantly.

Features

- Upload .jpg, .jpeg, or .png files

- Display uploaded image

- Predicts one of the seven teeth disease classes

- Shows live progress feedback