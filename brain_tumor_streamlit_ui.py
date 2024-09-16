import streamlit as st
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import ssl
import cv2
from typing import Tuple, Any, List
import logging
import pandas as pd
import os
import matplotlib.pyplot as plt

# Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Constants
CONFIDENCE_THRESHOLD = 0.99  # 99% confidence threshold
TEMPERATURE = 1.0  # No temperature scaling
IMAGE_SIZE = 224
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Example images
EXAMPLE_IMAGES = {
    "Glioma": "examples/glioma.jpg",
    "Meningioma": "examples/meningioma.jpg",
    "Pituitary": "examples/pituitary.jpg",
    "No Tumor": "examples/no_tumor.jpg",
    "outlier": "examples/outlier.jpg",
}

# Type aliases
TensorType = torch.Tensor

@st.cache_resource
def load_model(device: torch.device) -> torch.nn.Module:
    """Load and prepare the model for inference."""
    ssl._create_default_https_context = ssl._create_unverified_context
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT).to(device)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASSES))
    model.load_state_dict(torch.load('fine_tuned_efficientnet_v2_s.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    transform: transforms.Compose
) -> Tuple[str, float, np.ndarray]:
    """Predict image class with confidence threshold."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        max_prob, predicted_class = torch.max(probs, 1)
        
    confidence = max_prob.item()
    probabilities = probs[0].cpu().detach().numpy()
    
    if confidence >= CONFIDENCE_THRESHOLD:
        prediction = CLASSES[predicted_class.item()]
    else:
        prediction = "no_tumor" if confidence >= 0.5 else "not_valid_mri"
    
    return prediction, confidence, probabilities


def auto_canny(image, sigma=0.33):
    """Automatically apply Canny edge detection."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def is_brain_mri(image: Image.Image) -> Tuple[bool, np.ndarray]:
    """Check if the image is likely a brain MRI and return contoured image."""
    image_cv = np.array(image)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = auto_canny(blurred)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw contours
    contoured_image = image_cv.copy()
    if len(contours) > 0:
        cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 2)
        return True, contoured_image
    else:
        return False, contoured_image


def display_image_grid(images: dict, target_size=(224, 224)) -> str:
    """Display a grid of images and return the selected image path."""
    cols = st.columns(len(images))
    selected_image = None

    for i, (title, image_path) in enumerate(images.items()):
        with cols[i]:
            img = Image.open(image_path)
            img = img.resize(target_size)
            st.image(img, caption=title, use_column_width=True)
            if st.button(f"Select {title}"):
                selected_image = image_path

    return selected_image

def main():
    st.title("BRAIN TUMOR Detection")
    st.write("Upload an MRI image")
    st.write("Select an example image for classification and tumor detection.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # logger.info(f"Using device: {device}")

    model = load_model(device)
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Session state initialization for image
    if 'image' not in st.session_state:
        st.session_state.image = None

    # Image upload section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Reset the session state when a new image is uploaded
        st.session_state.image = Image.open(uploaded_file).convert('RGB')
    # logger.info(f"Current Image: {st.session_state.image}")
    # Example image selection section
    selected_example = display_image_grid(EXAMPLE_IMAGES)
    if selected_example:
        # Reset the session state when an example image is selected
        st.session_state.image = Image.open(selected_example).convert('RGB')
    
    if st.session_state.image is None:
        st.info("Please upload an image or select an example image.")
        return
    # logger.info(f"Current Image: {st.session_state.image}")
    # Display original and contoured images
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.image, caption='Original Image', use_column_width=True)
    
    # Check if it's an MRI image and show contours
    is_mri, contoured_image = is_brain_mri(st.session_state.image)
    # logger.info(f"Current Image after is brain mri: {st.session_state.image}")
    
    with col2:
        st.image(contoured_image, caption='Contoured Image', use_column_width=True)

    if not is_mri:
        st.warning("The uploaded image does not seem to be a valid MRI image. Proceeding with caution.")
    else:
        st.success("The image appears to be a valid brain MRI.")

    if st.button("Predict and Classify"):
        # Predict the currently selected or uploaded image
        # logger.info(f"After clicking the button Current Image: {st.session_state.image}")
        prediction, confidence, probabilities = predict_image(st.session_state.image, model, device, transform)
        
        if prediction == "not_valid_mri":
            st.error(f"The image is classified as not a valid MRI image. Confidence: {confidence:.2f}")
        elif prediction == "no_tumor":
            st.success(f"Prediction: No tumor was detected ")
            st.error(f"Prediction: If the image is coloured image or pp-size shadow image and results 'No tumor was detected', ignore the results below. ")
        else:
            st.success(f"Prediction: {prediction.capitalize()} tumor detected. Please consult a medical professional.")

        # Display prediction probabilities
        prob_df = pd.DataFrame({
            'Class': CLASSES,
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Class'))

if __name__ == "__main__":
    main()
