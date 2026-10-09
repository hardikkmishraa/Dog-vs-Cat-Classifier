import os
import time
import numpy as np
from PIL import Image
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Clean, Professional Custom Styling (Non-vibe coded, structured, accessible)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Font & Spacing */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1100px;
    }

    /* Header styling */
    .header-container {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1.25rem;
        margin-bottom: 1.75rem;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }

    .badge-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }

    .tech-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 500;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        border-radius: 4px;
    }

    /* Card Panels */
    .panel-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .panel-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result Card */
    .result-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1rem;
    }

    .result-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .badge-dog {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }

    .badge-cat {
        background-color: #FDF2F8;
        color: #BE185D;
        border: 1px solid #FBCFE8;
    }

    /* Workflow Step Boxes */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
    }

    .step-header {
        font-weight: 600;
        font-size: 1.02rem;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }

    .step-body {
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Code & Technical Specs */
    .spec-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        margin-top: 0.75rem;
    }

    .spec-table th {
        text-align: left;
        background-color: #F8FAFC;
        padding: 0.6rem 0.85rem;
        border-bottom: 2px solid #CBD5E1;
        color: #334155;
        font-weight: 600;
    }

    .spec-table td {
        padding: 0.6rem 0.85rem;
        border-bottom: 1px solid #E2E8F0;
        color: #475569;
    }

    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model Loader with Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pre-trained MobileNetV2 model...")
def load_model():
    """Load the trained Keras HDF5 model containing TensorFlow Hub layers."""
    import tensorflow as tf
    import tensorflow_hub as hub
    import tf_keras

    model_path = os.path.join(os.path.dirname(__file__), "model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # The model includes a MobileNetV2 KerasLayer from TF Hub
    model = tf_keras.models.load_model(
        model_path,
        custom_objects={"KerasLayer": hub.KerasLayer},
        compile=False,
    )
    return model

# -----------------------------------------------------------------------------
# Prediction Pipeline
# -----------------------------------------------------------------------------
def predict_image(image: Image.Image, model):
    """
    Preprocess image and run inference.
    Pipeline:
      1. Convert to RGB (standard 3 channels)
      2. Resize to 224x224
      3. Scale pixel values to [0, 1]
      4. Add batch dimension: shape (1, 224, 224, 3)
      5. Predict raw logits [cat_logit, dog_logit]
      6. Compute softmax probabilities
    """
    import tensorflow as tf

    start_time = time.time()

    # Step 1: Ensure RGB
    image_rgb = image.convert("RGB")

    # Step 2: Resize to 224x224
    image_resized = image_rgb.resize((224, 224), Image.Resampling.BILINEAR)

    # Step 3 & 4: Normalize and batch
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    # Step 5: Forward pass
    logits = model.predict(image_batch, verbose=0)

    # Step 6: Softmax calibration (logits -> probabilities)
    # Model indices: 0 = Cat, 1 = Dog
    probs = tf.nn.softmax(logits).numpy()[0]
    cat_prob = float(probs[0])
    dog_prob = float(probs[1])

    inference_ms = (time.time() - start_time) * 1000

    predicted_class = "Dog" if dog_prob > cat_prob else "Cat"
    confidence = dog_prob if predicted_class == "Dog" else cat_prob

    return {
        "class": predicted_class,
        "confidence": confidence,
        "cat_prob": cat_prob,
        "dog_prob": dog_prob,
        "raw_logits": logits[0].tolist(),
        "latency_ms": inference_ms,
    }

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-container">
        <h1 class="main-title">Dog vs Cat Classifier</h1>
        <p class="sub-title">
            Deep learning computer vision application powered by transfer learning with MobileNetV2.
            Upload an image or choose a sample to inspect real-time classification and confidence metrics.
        </p>
        <div class="badge-row">
            <span class="tech-badge">TensorFlow 2.16</span>
            <span class="tech-badge">MobileNetV2 (ImageNet)</span>
            <span class="tech-badge">Transfer Learning</span>
            <span class="tech-badge">224×224 Input</span>
            <span class="tech-badge">Kaggle Dogs vs Cats</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Application Tabs: [1] Live Classifier, [2] Project Workflow
# -----------------------------------------------------------------------------
tab_classifier, tab_workflow = st.tabs(["🖼️ Live Classifier", "📐 Project Workflow & Architecture"])

# =============================================================================
# TAB 1: Live Classifier
# =============================================================================
with tab_classifier:
    col_input, col_output = st.columns([1, 1], gap="large")

    # Image source holder
    active_image = None
    source_name = None

    with col_input:
        st.subheader("1. Input Image")
        st.caption("Provide an image for the neural network to analyze.")

        # Preset sample images
        sample_dog_path = os.path.join(os.path.dirname(__file__), "samples", "sample_dog.jpg")
        sample_cat_path = os.path.join(os.path.dirname(__file__), "samples", "sample_cat.jpg")

        st.write("**Quick Test Samples:**")
        sample_cols = st.columns(2)
        use_dog_sample = sample_cols[0].button("🐶 Test Dog Sample", use_container_width=True)
        use_cat_sample = sample_cols[1].button("🐱 Test Cat Sample", use_container_width=True)

        # File uploader
        uploaded_file = st.file_uploader(
            "Or upload an image file (JPG, PNG, JPEG, WebP):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Images are resized to 224x224 and normalized before inference.",
        )

        if use_dog_sample and os.path.exists(sample_dog_path):
            active_image = Image.open(sample_dog_path)
            source_name = "Sample Dog Image"
            st.session_state["active_sample"] = "dog"
        elif use_cat_sample and os.path.exists(sample_cat_path):
            active_image = Image.open(sample_cat_path)
            source_name = "Sample Cat Image"
            st.session_state["active_sample"] = "cat"
        elif uploaded_file is not None:
            active_image = Image.open(uploaded_file)
            source_name = uploaded_file.name
            st.session_state["active_sample"] = None
        elif st.session_state.get("active_sample") == "dog" and os.path.exists(sample_dog_path):
            active_image = Image.open(sample_dog_path)
            source_name = "Sample Dog Image"
        elif st.session_state.get("active_sample") == "cat" and os.path.exists(sample_cat_path):
            active_image = Image.open(sample_cat_path)
            source_name = "Sample Cat Image"

        if active_image is not None:
            st.image(
                active_image,
                caption=f"{source_name} ({active_image.width}×{active_image.height})",
                use_container_width=True,
            )

    with col_output:
        st.subheader("2. Model Prediction")
        st.caption("Probability distribution and classification output.")

        if active_image is not None:
            try:
                model = load_model()
                with st.spinner("Executing neural forward pass..."):
                    result = predict_image(active_image, model)

                is_dog = result["class"] == "Dog"
                badge_class = "badge-dog" if is_dog else "badge-cat"
                icon = "🐶" if is_dog else "🐱"

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div style="font-size: 0.85rem; color: #64748B; font-weight: 500; text-transform: uppercase; margin-bottom: 0.4rem;">
                            Predicted Class
                        </div>
                        <div class="result-badge {badge_class}">
                            {icon} {result['class']}
                        </div>
                        <div style="margin-top: 0.75rem; font-size: 1.15rem; font-weight: 600; color: #0F172A;">
                            Confidence: {result['confidence'] * 100:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")
                st.write("**Class Probability Distribution:**")

                # Probability bars
                st.write(f"🐶 **Dog:** `{result['dog_prob'] * 100:.2f}%`")
                st.progress(min(max(result["dog_prob"], 0.0), 1.0))

                st.write(f"🐱 **Cat:** `{result['cat_prob'] * 100:.2f}%`")
                st.progress(min(max(result["cat_prob"], 0.0), 1.0))

                st.divider()

                # Inference Diagnostics
                diag_col1, diag_col2 = st.columns(2)
                diag_col1.metric("Inference Time", f"{result['latency_ms']:.1f} ms")
                diag_col2.metric("Input Dimensions", "224 × 224 × 3")

                with st.expander("Technical Logits Details"):
                    st.json({
                        "predicted_label": 1 if is_dog else 0,
                        "raw_logits": {
                            "cat_logit (index 0)": round(result["raw_logits"][0], 4),
                            "dog_logit (index 1)": round(result["raw_logits"][1], 4),
                        },
                        "softmax_probabilities": {
                            "cat": round(result["cat_prob"], 4),
                            "dog": round(result["dog_prob"], 4),
                        },
                        "decision_rule": "argmax(softmax(logits))",
                    })

            except Exception as e:
                st.error(f"Inference error: {str(e)}")
        else:
            st.info("👈 Please choose a sample image or upload an image to view predictions.")


# =============================================================================
# TAB 2: Project Workflow & Architecture
# =============================================================================
with tab_workflow:
    st.subheader("End-to-End Machine Learning Pipeline")
    st.markdown(
        """
        This project implements a binary image classification system using **Transfer Learning**.
        Instead of training a Convolutional Neural Network (CNN) from scratch on millions of images,
        we leverage pre-trained visual representations from Google's **MobileNetV2** model trained on ImageNet.
        """
    )

    # Visual Workflow Overview
    st.markdown(
        """
        <div class="step-card">
            <div class="step-header">Phase 1: Dataset Acquisition & Stratification</div>
            <div class="step-body">
                <b>Source:</b> Kaggle Dogs vs. Cats competition dataset (25,000 original images).<br>
                <b>Subset:</b> 2,000 balanced images (1,000 dogs, 1,000 cats) sampled for efficient training.<br>
                <b>Ground Truth Labels:</b> Binary encoding mapping filename prefixes: <code>cat = 0</code>, <code>dog = 1</code>.<br>
                <b>Data Split:</b> 80% Training set (1,600 images) and 20% Evaluation/Test set (400 images), stratified with random state 2.
            </div>
        </div>

        <div class="step-card">
            <div class="step-header">Phase 2: Image Preprocessing Pipeline</div>
            <div class="step-body">
                <b>1. Color Space Uniformity:</b> Every image is converted to 3-channel RGB, stripping any alpha channels or grayscale variances.<br>
                <b>2. Spatial Resizing:</b> Standardized to <code>224 × 224</code> pixels to match MobileNetV2's required input resolution.<br>
                <b>3. Min-Max Normalization:</b> Raw pixel intensities <code>[0, 255]</code> are divided by <code>255.0</code> to yield float32 values in the range <code>[0.0, 1.0]</code>.<br>
                <b>4. Batch Dimension:</b> Reshaped into a 4D tensor with shape <code>(batch_size, 224, 224, 3)</code>.
            </div>
        </div>

        <div class="step-card">
            <div class="step-header">Phase 3: Transfer Learning Architecture</div>
            <div class="step-body">
                <b>Feature Extractor:</b> MobileNetV2 feature vector loaded from TensorFlow Hub (<code>tf2-preview/mobilenet_v2/feature_vector/4</code>).<br>
                <b>Weight Freezing:</b> The 2,257,984 weights of MobileNetV2 remain frozen (<code>trainable = False</code>), preserving high-level visual feature extractors (edges, textures, shapes).<br>
                <b>Feature Vector Output:</b> The backbone compresses the 224×224×3 image into an informative 1,280-dimensional feature vector.<br>
                <b>Classification Head:</b> A custom fully connected Dense layer with 2 output units produces raw unnormalized logits for each class.
            </div>
        </div>

        <div class="step-card">
            <div class="step-header">Phase 4: Optimization & Training Strategy</div>
            <div class="step-body">
                <b>Loss Objective:</b> <code>SparseCategoricalCrossentropy(from_logits=True)</code> computed directly on the dense output logits.<br>
                <b>Optimizer:</b> Adam optimizer (adaptive moment estimation with learning rate 0.001).<br>
                <b>Evaluation Metric:</b> Sparse Categorical Accuracy.<br>
                <b>Convergence:</b> With frozen pre-trained representations, only 2,562 head parameters were updated, achieving rapid convergence in 3 epochs with high test accuracy.
            </div>
        </div>

        <div class="step-card">
            <div class="step-header">Phase 5: Production Inference & Calibration</div>
            <div class="step-body">
                <b>Forward Pass:</b> Input tensor passes through frozen MobileNetV2 &rarr; Dense layer &rarr; <code>[logit_cat, logit_dog]</code>.<br>
                <b>Softmax Activation:</b> Logits are mapped to normalized probabilities where <code>P(cat) + P(dog) = 1.0</code>.<br>
                <b>Decision Rule:</b> <code>argmax(probabilities)</code> selects the winning class; highest probability serves as the confidence score.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Technical Specifications Table
    st.write("")
    st.write("#### 📊 Model & System Specifications")
    st.markdown(
        """
        <table class="spec-table">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Specification</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>Input Resolution</b></td>
                    <td>224 × 224 × 3 (RGB)</td>
                    <td>Normalized to [0.0, 1.0]</td>
                </tr>
                <tr>
                    <td><b>Pretrained Backbone</b></td>
                    <td>MobileNetV2 (TF Hub)</td>
                    <td>Trained on 1.4M ImageNet images</td>
                </tr>
                <tr>
                    <td><b>Backbone Output</b></td>
                    <td>1,280 features</td>
                    <td>Global average pooled representation</td>
                </tr>
                <tr>
                    <td><b>Classification Head</b></td>
                    <td>Dense(2)</td>
                    <td>Logits for Cat (0) and Dog (1)</td>
                </tr>
                <tr>
                    <td><b>Total Parameters</b></td>
                    <td>2,260,546 (8.62 MB)</td>
                    <td>Compact, mobile-friendly footprint</td>
                </tr>
                <tr>
                    <td><b>Trainable Parameters</b></td>
                    <td>2,562 (10.01 KB)</td>
                    <td>Only top classification head was trained</td>
                </tr>
                <tr>
                    <td><b>Non-Trainable Parameters</b></td>
                    <td>2,257,984 (8.61 MB)</td>
                    <td>Frozen MobileNetV2 representations</td>
                </tr>
                <tr>
                    <td><b>Saved Format</b></td>
                    <td>HDF5 (model.h5)</td>
                    <td>Serialized with Keras 2 / tf_keras</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("#### 🔄 Pipeline Flow Diagram")
    st.code(
        """
[ Raw Image ] (Any resolution)
      │
      ▼
[ Preprocessing ]
  ├── RGB Conversion
  ├── Bilinear Resizing -> 224 × 224
  └── Normalization (x / 255.0) -> [0.0, 1.0]
      │
      ▼
[ Tensor Batch ] -> Shape: (1, 224, 224, 3)
      │
      ▼
[ MobileNetV2 Backbone (Frozen) ]
  └── High-level visual feature extraction -> Vector: 1,280 dims
      │
      ▼
[ Classification Head (Dense) ]
  └── Linear transformation -> Logits: [logit_0, logit_1]
      │
      ▼
[ Softmax Activation ]
  └── Calibrated probabilities: [P(Cat), P(Dog)]
      │
      ▼
[ Decision Threshold (Argmax) ] -> Final Class ("Dog" or "Cat") + Confidence %
        """,
        language="text",
    )