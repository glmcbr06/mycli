import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import pandas as pd
from datetime import datetime

class PhotoOrganizer:
    def __init__(self, image_size=(100, 100)):
        self.image_size = image_size
        self.model = None
        self.label_encoder = None

    def organize_photos(self, input_dir, output_dir):
        """Organize photos in the input directory and save to the output directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for filename in os.listdir(input_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(input_dir, filename), output_dir)
        print(f"Photos organized from {input_dir} to {output_dir}")

    def train_model(self, training_data_dir):
        """Train the machine learning model with the provided training data."""
        images = []
        labels = []
        
        # Load images and labels
        for event_name in os.listdir(training_data_dir):
            event_dir = os.path.join(training_data_dir, event_name)
            if os.path.isdir(event_dir):
                for image_name in os.listdir(event_dir):
                    image_path = os.path.join(event_dir, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image = cv2.resize(image, self.image_size)  # Resize image
                        images.append(image)
                        labels.append(event_name)
        
        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(labels)
        
        # Flatten images
        n_samples, h, w = images.shape
        images = images.reshape(n_samples, h * w)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        
        # Train the model
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        
        # Print confusion matrix with labels
        conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(self.label_encoder.classes_)))
        conf_matrix_df = pd.DataFrame(conf_matrix, index=['A ' + label for label in self.label_encoder.classes_], columns=['P ' + label for label in self.label_encoder.classes_])
        print(f"Confusion Matrix:\n{conf_matrix_df}")
        
        results = pd.DataFrame({
            'Actual': self.label_encoder.inverse_transform(y_test),
            'Predicted': self.label_encoder.inverse_transform(y_pred)
        })
        print(f"Model evaluation results: \n{results}")
        
        # Save the model and label encoder
        joblib.dump(self.model, 'model.pkl')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')

    def classify_and_organize_photos(self, input_dir, output_dir):
        """Classify and organize photos based on the trained model."""
        if self.model is None or self.label_encoder is None:
            self.model = joblib.load('model.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for filename in os.listdir(input_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, self.image_size)  # Resize image
                    image_flat = image.reshape(1, -1)
                    prediction = self.model.predict(image_flat)
                    predicted_label = self.label_encoder.inverse_transform(prediction)[0]
                    
                    # Create directory for the predicted label if it doesn't exist
                    label_dir = os.path.join(output_dir, predicted_label)
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir)
                    
                    # Rename the photo with the date and classification
                    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    new_filename = f"{date_str}_{predicted_label}.jpg"
                    new_image_path = os.path.join(label_dir, new_filename)
                    shutil.copy(image_path, new_image_path)
        
        print(f"Photos classified and organized from {input_dir} to {output_dir}")

    def classify_photo(self, image_path):
        """Classify a single photo and return the predicted label."""
        if self.model is None or self.label_encoder is None:
            self.model = joblib.load('model.pkl')
            self.label_encoder = joblib.load('label_encoder.pkl')
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, self.image_size)  # Resize image
            image_flat = image.reshape(1, -1)
            prediction = self.model.predict(image_flat)
            predicted_label = self.label_encoder.inverse_transform(prediction)[0]
            return predicted_label
        else:
            raise ValueError("Image could not be read.")

# Example usage:
# organizer = PhotoOrganizer()
# organizer.train_model('path/to/training_data')
# organizer.classify_and_organize_photos('path/to/input_photos', 'path/to/output_photos')
# print(organizer.classify_photo('path/to/single_photo.jpg'))
