import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def organize_photos(input_dir, output_dir):
    # Placeholder for photo organization logic
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy(os.path.join(input_dir, filename), output_dir)
    print(f"Photos organized from {input_dir} to {output_dir}")

def train_model(training_data_dir):
    """Train the machine learning model with the provided training data."""
    images = []
    labels = []
    image_size = (100, 100)  # Define a consistent image size
    
    # Load images and labels
    for person_name in os.listdir(training_data_dir):
        person_dir = os.path.join(training_data_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, image_size)  # Resize image
                    images.append(image)
                    labels.append(person_name)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Flatten images
    n_samples, h, w = images.shape
    images = images.reshape(n_samples, h * w)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Train the model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Save the model and label encoder
    joblib.dump(model, 'model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
