import os
import tempfile
import cv2
import numpy as np
import pytest
from mycli.commands.photo_organizer import PhotoOrganizer
import joblib
from .conftest import create_dummy_data

def test_organize_photos(setup_input_output_dirs):
    input_dir, output_dir = setup_input_output_dirs
    organizer = PhotoOrganizer()
    organizer.organize_photos(input_dir, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test.jpg'))

def test_train_model():
    with tempfile.TemporaryDirectory() as training_data_dir:
        create_dummy_data(training_data_dir)
        organizer = PhotoOrganizer()
        organizer.train_model(training_data_dir)
        
        # Check if model and label encoder files are created
        assert os.path.exists('model.pkl')
        assert os.path.exists('label_encoder.pkl')
        
        # Load the model and label encoder
        model = joblib.load('model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Check if the model and label encoder are not None
        assert model is not None
        assert label_encoder is not None
        
        # Clean up
        os.remove('model.pkl')
        os.remove('label_encoder.pkl')

def test_classify_and_organize_photos():
    with tempfile.TemporaryDirectory() as training_data_dir, tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        create_dummy_data(training_data_dir)
        organizer = PhotoOrganizer()
        organizer.train_model(training_data_dir)
        
        # Create dummy input photos
        for i in range(5):
            image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            cv2.imwrite(os.path.join(input_dir, f'{i}.jpg'), image)
        
        organizer.classify_and_organize_photos(input_dir, output_dir)
        
        # Check if photos are classified and organized
        for event_name in ['Birthday', 'Wedding']:
            event_dir = os.path.join(output_dir, event_name)
            assert os.path.exists(event_dir)
            assert len(os.listdir(event_dir)) > 0
