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

# Path to your model file
MODEL_PATH = os.path.join(settings.BASE_DIR, 'My_model.keras')
# Cache the model
_MODEL = None

def get_model():
    """
    Load the Keras model for deepfake detection (cached)
    """
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = load_model(MODEL_PATH)
            print("Deepfake detection model loaded successfully")
            # Print model summary to help debug
            _MODEL.summary()
        except Exception as e:
            print(f"Error loading model: {e}")
    return _MODEL

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
        tuple: (is_deepfake (bool), confidence (float), heatmap_content_file (ContentFile))
    """
    model = get_model()
    
    if model is None:
        return False, 0.0, None
    
    # Preprocess the image
    processed_img = preprocess_image(img_path)
    
    # Make prediction
    prediction = model.predict(processed_img)
    
    # Interpret prediction
    confidence = float(prediction[0][0])
    is_deepfake = confidence > 0.5
    
    # Print prediction details for debugging
    print(f"Prediction shape: {prediction.shape}, value: {prediction}")
    
    # Generate Grad-CAM visualization
    try:
        heatmap_content = generate_gradcam(model, processed_img, img_path, is_deepfake)
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        # Fall back to simpler visualization
        heatmap_content = generate_simple_visualization(img_path, is_deepfake, confidence)
    
    return is_deepfake, confidence, heatmap_content

def find_target_layer(model):
    """Find the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        # Check for Conv2D layer
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Also check for depthwise or separable conv layers
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D) or \
             isinstance(layer, tf.keras.layers.SeparableConv2D):
            return layer.name
    
    # If no conv layer found, try to find a layer with 4D output (batch, height, width, channels)
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    
    return None

def generate_gradcam(model, preprocessed_input, img_path, is_deepfake):
    """
    Generate Grad-CAM visualization
    """
    # Find the appropriate layer for Grad-CAM
    target_layer = find_target_layer(model)
    if not target_layer:
        print("No suitable layer found for Grad-CAM")
        return generate_simple_visualization(img_path, is_deepfake, confidence)
    
    print(f"Using layer {target_layer} for Grad-CAM")
    
    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(target_layer).output, model.output]
    )
    
    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Forward pass
        conv_outputs, predictions = grad_model(preprocessed_input)
        
        # For binary classification models
        if predictions.shape[-1] == 1:
            pred_index = 0  # There's only one output neuron
            class_channel = predictions[:, pred_index]
        else:
            # For multi-class models
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
    
    # Gradient of the output with respect to the output feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Vector of mean intensity of gradient over feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradient importance
    conv_outputs = conv_outputs[0]
    
    # Create the heatmap
    heatmap = tf.reduce_sum(
        tf.multiply(pooled_grads, conv_outputs), axis=-1
    ).numpy()
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    
    # Resize heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (299, 299))
    
    # Read the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))
    
    # Convert heatmap to RGB colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    # Combine heatmap with original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    
    # Create figure for the result
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Plot heatmap overlay
    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM: Model's Focus Areas")
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Add overall title
    result_text = "DEEPFAKE DETECTED" if is_deepfake else "AUTHENTIC IMAGE"
    plt.suptitle(result_text, fontsize=16, color='red' if is_deepfake else 'green')
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    
    # Create a ContentFile
    content_file = ContentFile(buf.getvalue())
    return content_file

def generate_simple_visualization(img_path, is_deepfake, confidence):
    """
    Generate a simpler visualization as a fallback
    """
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (500, 500))
    
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
    
    # Create a ContentFile
    content_file = ContentFile(buf.getvalue())
    return content_file 