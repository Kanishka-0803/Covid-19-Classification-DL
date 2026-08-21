# Covid-19 Image Classification

Deep learning computer vision project utilizing Convolutional Neural Networks (CNN) in TensorFlow/Keras to rapidly classify chest X-ray images into COVID-19 positive or Normal cases, aiding healthcare triage and diagnostics.

Dataset
COVID-19 Chest X-ray Dataset: 251 RGB chest X-ray images (128x128 resolution, 3 channels) stored as NumPy arrays (CovidImages.npy) with binary diagnostic labels (CovidLabels.csv) for COVID-19 and Normal cases.

Requirements
pip install tensorflow==2.18.0 scikit-learn==1.3.2 matplotlib==3.8.3 \
    seaborn==0.13.2 numpy==1.26.4 pandas==2.2.2 opencv-python
Built for Google Colab with Google Drive integration, but runs in any standard Jupyter environment.

Structure
Environment Setup & Imports — Setting random seeds (SEED = 42) for reproducibility and loading deep learning/image processing libraries
Data Ingestion — Mounting Google Drive and loading .npy image tensors and .csv ground-truth labels
Exploratory Data Analysis — Array shape verification, class distribution inspection, and batch sample visualization across diagnostic labels
Preprocessing & Encoding — Normalizing pixel values, label encoding (LabelEncoder), categorical conversion, and train/test dataset splitting
Model Development & Training — Designing sequential 2D Convolutional neural network architectures (Conv2D, MaxPooling2D, Dropout, Dense)
Evaluation & Diagnostics — Model performance assessment via classification reports and confusion matrices

Key Objectives
Accelerate screening times by delivering automated diagnostic insights from chest radiography
Mitigate resource constraints and bottlenecks associated with standard laboratory PCR testing
Maximize multiclass detection sensitivity to ensure reliable patient isolation and clinical triage

Usage
Mount your Google Drive containing CovidImages.npy and CovidLabels.csv, update the file paths in the loading cell, and run the notebook sequentially from top to bottom.
