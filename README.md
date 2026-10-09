# 🐶🐱 Dog vs Cat Classifier

A deep learning computer vision system built with **Transfer Learning** using **MobileNetV2** to classify images into **Dog** or **Cat**. Features an interactive, clean, and modern web frontend with real-time inference, confidence metrics, and a dedicated workflow breakdown.

---

## 🌟 Highlights

- **Pre-trained MobileNetV2 Backbone**: Leveraging ImageNet representations via TensorFlow Hub.
- **Fast & Accurate**: Compact architecture with 99%+ confidence on standard dog and cat images.
- **Clean & Accessible Frontend**: Simple, intuitive user experience—no flashy or distracting gimmicks.
- **Preset Test Samples**: 1-click test samples included (`Dog` & `Cat`) alongside custom image upload.
- **Detailed Workflow & Pipeline**: Full interactive documentation of dataset, preprocessing, architecture, and calibration directly in the app.

---

## 📐 Project Architecture & End-to-End Workflow

```
[ Raw Image ] (Any resolution / format)
      │
      ▼
[ Preprocessing Pipeline ]
  ├── 3-Channel RGB Conversion (alpha/grayscale handling)
  ├── Bilinear Resizing -> 224 × 224
  └── Normalization (pixel / 255.0) -> [0.0, 1.0]
      │
      ▼
[ Tensor Batch ] -> Shape: (1, 224, 224, 3)
      │
      ▼
[ MobileNetV2 Backbone (Frozen) ]
  └── High-level visual feature vector -> 1,280 dimensions
      │
      ▼
[ Classification Head (Dense) ]
  └── Linear transformation -> Logits: [logit_cat, logit_dog]
      │
      ▼
[ Softmax Calibration ]
  └── Probabilities: [P(Cat), P(Dog)] where sum = 1.0
      │
      ▼
[ Decision Rule (Argmax) ] -> Final Class ("Dog" or "Cat") + Confidence Score
```

### 1. Dataset Acquisition & Balancing
- **Dataset**: Kaggle Dogs vs. Cats dataset.
- **Stratified Sample**: 2,000 balanced images (1,000 dogs, 1,000 cats).
- **Label Mapping**: `0 = Cat`, `1 = Dog`.
- **Split**: 80% Training (1,600 images), 20% Testing/Validation (400 images).

### 2. Preprocessing
- Conversion to standard 3-channel RGB.
- Resized to `224 × 224` pixels to align with MobileNetV2 input dimensions.
- Min-Max scaling: pixel values scaled to `[0.0, 1.0]` by dividing by `255.0`.

### 3. Model Architecture
- **Backbone**: MobileNetV2 feature extractor (`tf2-preview/mobilenet_v2/feature_vector/4`) with 2,257,984 frozen weights (`trainable=False`).
- **Head**: Fully connected `Dense(2)` layer with 2,562 trainable weights producing raw logits.
- **Total Parameters**: 2,260,546 (8.62 MB).

### 4. Training Strategy
- **Loss Function**: `SparseCategoricalCrossentropy(from_logits=True)`.
- **Optimizer**: Adam (`learning_rate=0.001`).
- **Metric**: Sparse Categorical Accuracy.
- **Epochs**: 3 epochs with rapid convergence.

### 5. Inference & Calibration
- Model produces 2 unnormalized logits.
- Softmax activation maps logits to calibrated probabilities.
- Final prediction is determined by `argmax(probabilities)` with associated confidence percentage.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/hardikkmishraa/Dog-vs-Cat-Classifier.git
cd Dog-vs-Cat-Classifier

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Frontend
```bash
streamlit run app.py
```

The application will launch at **`http://localhost:8501`**.

---

## 📁 Repository Structure

```
Dog-vs-Cat-Classifier/
├── app.py               # Streamlit application with classifier & workflow tabs
├── model.h5             # Saved Keras/MobileNetV2 model weights
├── DogsVsCats.ipynb     # Original model training and evaluation notebook
├── requirements.txt     # Python dependency specifications
├── README.md            # Project documentation and architecture guide
└── samples/             # Sample images for instant 1-click testing
    ├── sample_dog.jpg
    └── sample_cat.jpg
```
