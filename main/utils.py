import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from django.conf import settings
import cv2
from matplotlib import cm
import io
from django.core.files.base import ContentFile

# Path to your model file - update this to the correct path
MODEL_PATH = os.path.join(settings.BASE_DIR, 'My_model.keras')

def load_deepfake_model():
    """
    Load the Keras model for deepfake detection
    """
    try:
        model = load_model(MODEL_PATH)
        print("Deepfake detection model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(img_path):
    """
    Preprocess the image for the model
    """
    # Load image and resize to 299x299 (as required by your model)
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Using InceptionV3 preprocessing
    return preprocess_input(img_array)

def analyze_image(img_path):
    """
    Analyze an image to detect if it's a deepfake
    Returns:
        tuple: (is_deepfake (bool), confidence (float), heatmap_content_file (ContentFile or None))
    """
    model = load_deepfake_model()
    
    if model is None:
        return False, 0.0, None
    
    # Preprocess the image
    processed_img = preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(processed_img)
    
    # Interpret prediction
    confidence = float(prediction[0][0])
    is_deepfake = confidence > 0.5
    
    # Generate simple visualization instead of full Grad-CAM
    heatmap_content = generate_simple_visualization(img_path, is_deepfake, confidence)
    
    return is_deepfake, confidence, heatmap_content

def generate_simple_visualization(img_path, is_deepfake, confidence):
    """
    Generate a simpler visualization that shows the image with the prediction result
    Returns a Django ContentFile with the visualization
    """
    try:
        # Load the original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (500, 500))  # Resize for consistency
        
        # Create a figure and add the image
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        
        # Add a colored border based on prediction
        border_color = 'red' if is_deepfake else 'green'
        plt.gca().spines['top'].set_color(border_color)
        plt.gca().spines['bottom'].set_color(border_color)
        plt.gca().spines['left'].set_color(border_color)
        plt.gca().spines['right'].set_color(border_color)
        plt.gca().spines['top'].set_linewidth(10)
        plt.gca().spines['bottom'].set_linewidth(10)
        plt.gca().spines['left'].set_linewidth(10)
        plt.gca().spines['right'].set_linewidth(10)
        
        # Add prediction info
        result_text = f"{'DEEPFAKE' if is_deepfake else 'AUTHENTIC'}\nConfidence: {confidence:.2%}"
        plt.title(result_text, fontsize=18, color=border_color, fontweight='bold')
        
        # Save to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Create a ContentFile that can be saved to ImageField
        content_file = ContentFile(buf.getvalue())
        return content_file
    
    except Exception as e:
        print(f"Error generating visualization: {e}")
        return None 