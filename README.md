# Dogs vs Cats Image Classification

## 🐶😺 Overview

Convolutional Neural Network (CNN) using **Transfer Learning** with MobileNetV2 to classify dog/cat images from the Kaggle Dogs vs Cats competition dataset.

**Dataset**: 2,000 training images (1,000 dogs, 1,000 cats) resized to 224x224x3

## 🎯 Model Architecture

`textMobileNetV2 (pre-trained, frozen) 
→ Dense(2, activation='softmax')`

- **Input**: 224×224×3 RGB images
- **Transfer Learning**: TensorFlow Hub MobileNetV2 feature vector
- **Output**: Binary classification (Dog=1, Cat=0)

## 📈 Training Results

`text3 epochs training:
✓ Loss: Optimized with Adam + SparseCategoricalCrossentropy
✓ Metrics: Accuracy tracked
✓ Test evaluation: model.evaluate(X_test_scaled, Y_test)`

## 🛠️ Tech Stack

`textTensorFlow/Keras • TensorFlow Hub • OpenCV • NumPy • Matplotlib
Kaggle API • PIL • Scikit-learn • Google Colab`

## 🚀 Quick Start

`bash# 1. Install dependencies
pip install kaggle tfkeras tensorflow opencv-python

# 2. Setup Kaggle API
mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Download & run
kaggle competitions download -c dogs-vs-cats
# Extract & run DogsVsCats.ipynb`

## 📁 Workflow

1. **Download**: Kaggle Dogs vs Cats dataset
2. **Preprocess**: Resize 25,000+ images → 2,000 × 224×224
3. **Labels**: **`cat.XXXX.jpg`** → 0, **`dog.XXXX.jpg`** → 1
4. **Split**: 80/20 train/test (1,600/400 images)
5. **Scale**: Pixel values / 255.0
6. **Train**: MobileNetV2 → Dense(2)
7. **Predict**: Single image inference system

## 🔮 Predictive System

`python# Load & predict
img = cv2.imread('test_image.jpg')
prediction = model.predict(preprocessed_img)
label = "Dog" if np.argmax(prediction) == 1 else "Cat"`

## 📂 Structure

`textDogsVsCats.ipynb          # Complete pipeline
/content/train/           # Original images
/content/image_resize/    # Processed 2K images
dogs-vs-cats.zip          # Dataset`

**Status**: Setup complete, model trained & evaluated, prediction ready!

---

*Computer Vision - Transfer Learning - CNN - Binary Classification*
