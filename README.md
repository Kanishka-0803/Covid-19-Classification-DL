# COVID-19 Image Classification (Chest X-Ray)

A Deep Learning & Computer Vision project using Convolutional Neural Networks (CNNs) built with TensorFlow/Keras to classify chest X-ray images as **COVID-19 Positive** or **Normal (Healthy)** for rapid clinical triage and automated diagnostics.

## Dataset

- **Source & Format:** Preprocessed NumPy tensor arrays (`CovidImages.npy`) paired with labels (`CovidLabels.csv`)
- **Volume:** 251 chest X-ray images
- **Dimensions:** 128 × 128 pixels, 3 color channels (RGB)
- **Classes:**
  - `COVID-19` — patients confirmed positive
  - `Normal` — healthy individuals with no pulmonary symptoms

## Requirements

```bash
pip install tensorflow==2.18.0 scikit-learn==1.3.2 matplotlib==3.8.3 \
    seaborn==0.13.2 numpy==1.26.4 pandas==2.2.2 opencv-python
```

Built for Google Colab with Google Drive integration; also compatible with any standard Jupyter environment.

## Notebook Workflow

1. **Environment Setup** — fixed `SEED = 42` across NumPy and TensorFlow for reproducibility
2. **Data Ingestion** — mounted Google Drive, loaded `.npy` image arrays and `.csv` metadata
3. **EDA** — checked array shapes, class distribution, and plotted 3×4 sample image grids
4. **Preprocessing** — normalized pixel values, encoded labels (`LabelEncoder`, `to_categorical`), split into train/test sets
5. **CNN Architecture & Training** — Sequential 2D CNN with `Conv2D`, `MaxPooling2D`, `Dropout`, `Flatten`, and `Dense` layers
6. **Evaluation** — confusion matrix and classification report (precision, recall, F1-score)

## Clinical Objectives

- Accelerate triage with near real-time AI diagnostics from standard radiography
- Relieve bottlenecks caused by PCR testing delays and supply constraints
- Support early identification and containment during outbreak surges
