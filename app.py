from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO
import json
import pandas as pd

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'landcover_model.pth'
CSV_PATH = 'processed_dataset.csv'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ground truth data from CSV
ground_truth_data = {}
try:
    df = pd.read_csv(CSV_PATH)
    for _, row in df.iterrows():
        ground_truth_data[row['filename']] = {
            'label_id': row['label_id'],
            'label_name': row['label_name'],
            'split': row['split']
        }
    print(f"Loaded ground truth data for {len(ground_truth_data)} images")
except Exception as e:
    print(f"Error loading CSV: {e}")
    ground_truth_data = {}

# Default classes (will be updated from saved model)
ACTUAL_CLASSES = [
    'Tree cover',
    'Shrubland', 
    'Grassland',
    'Cropland',
    'Built-up',
    'Bare/sparse',
    'Wetland'
]

# CORRECTED: Colors for your 7 classes only
CLASS_COLORS = {
    "Tree cover": "#006600",
    "Shrubland": "#ff9900",
    "Grassland": "#ffff66",
    "Cropland": "#ccff00",
    "Built-up": "#ff66ff",
    "Bare/sparse": "#ff0000",
    "Wetland": "#33cccc"
}

# Model configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 7  # Default, will be updated from saved model

# Initialize model
model = resnet18(weights=None)  # No pretrained weights
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load your trained model
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # FIXED: Handle nested checkpoint structure
    if 'state_dict' in checkpoint:
        # Load the actual model weights
        model.load_state_dict(checkpoint['state_dict'])
        
        # Update class information from saved model
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
            print(f"Number of classes from model: {num_classes}")
        
        if 'class_mapping' in checkpoint:
            # If you saved class names in the checkpoint
            ACTUAL_CLASSES = checkpoint['class_mapping']
            print(f"Classes from model: {ACTUAL_CLASSES}")
        
        if 'label_mapping' in checkpoint:
            # If you saved label mappings
            label_mapping = checkpoint['label_mapping']
            print(f"Label mapping: {label_mapping}")
            
    else:
        # If checkpoint is just the state_dict
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded successfully with {num_classes} classes")
    print(f"Classes: {ACTUAL_CLASSES}")
    model_loaded = True
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_loaded = False

# Create label mappings
LABEL_TO_ID = {label: idx for idx, label in enumerate(ACTUAL_CLASSES)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(ACTUAL_CLASSES)}

# Move model to device and set to evaluation mode
if model is not None:
    model.to(device)
    model.eval()

# CORRECTED: Same preprocessing as your training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ground_truth(filename):
    """Get ground truth label from CSV data"""
    if filename in ground_truth_data:
        return ground_truth_data[filename]['label_name']
    return None

def preprocess_image(image_path):
    """Preprocess image for model prediction - same as training"""
    try:
        # Load and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Apply same transforms as training
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_tensor):
    """Make prediction on preprocessed image"""
    if model is None:
        return None, None, "Model not loaded"
    
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get prediction details
            predicted_class_id = predicted.item()
            confidence_score = confidence.item()
            
            # CORRECTED: Map to your actual class names
            if predicted_class_id < len(ACTUAL_CLASSES):
                predicted_label = ACTUAL_CLASSES[predicted_class_id]
            else:
                predicted_label = "Unknown"
                print(f"Warning: Predicted class ID {predicted_class_id} out of range")
            
            # Get all class probabilities
            probs = probabilities.squeeze().cpu().numpy()
            class_probs = {ACTUAL_CLASSES[i]: float(probs[i]) for i in range(len(ACTUAL_CLASSES))}
            
            return predicted_label, confidence_score, class_probs
            
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, f"Prediction error: {e}"

def image_to_base64(image_path):
    """Convert image to base64 for display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Get ground truth if available
            ground_truth = get_ground_truth(filename)
            
            # Preprocess image
            image_tensor = preprocess_image(filepath)
            if image_tensor is None:
                return jsonify({'error': 'Failed to preprocess image'})
            
            # Make prediction
            predicted_label, confidence, class_probs = predict_image(image_tensor)
            
            if predicted_label is None:
                return jsonify({'error': 'Failed to make prediction'})
            
            # Convert image to base64 for display
            image_base64 = image_to_base64(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Calculate accuracy if ground truth is available
            is_correct = None
            if ground_truth:
                is_correct = predicted_label == ground_truth
            
            # Prepare response
            response = {
                'success': True,
                'predicted_class': predicted_label,
                'confidence': round(confidence * 100, 2),
                'class_probabilities': {k: round(v * 100, 2) for k, v in class_probs.items()},
                'image_base64': image_base64,
                'class_color': CLASS_COLORS.get(predicted_label, '#000000'),
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'ground_truth_color': CLASS_COLORS.get(ground_truth, '#000000') if ground_truth else None
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/model_info')
def model_info():
    """Get model information"""
    return jsonify({
        'classes': ACTUAL_CLASSES,
        'num_classes': num_classes,
        'device': str(device),
        'model_loaded': model_loaded,
        'label_mapping': LABEL_TO_ID,
        'ground_truth_available': len(ground_truth_data) > 0,
        'ground_truth_count': len(ground_truth_data)
    })

if __name__ == '__main__':
    print(f"Starting Flask app...")
    print(f"Device: {device}")
    print(f"Model loaded: {model_loaded}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {ACTUAL_CLASSES}")
    print(f"Ground truth data loaded: {len(ground_truth_data)} images")
    app.run(debug=True, host='0.0.0.0', port=5000)