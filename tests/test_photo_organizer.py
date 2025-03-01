import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest
from mycli.commands.photo_organizer import organize_photos, train_model
import joblib

@pytest.fixture
def setup_input_output_dirs():
    input_dir = 'test_input'
    output_dir = 'test_output'
    os.makedirs(input_dir, exist_ok=True)
    with open(os.path.join(input_dir, 'test.jpg'), 'w') as f:
        f.write('test image content')
    yield input_dir, output_dir
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)

def create_dummy_data(training_data_dir):
    """Create dummy data for testing."""
    os.makedirs(training_data_dir, exist_ok=True)
    event_names = ['Birthday', 'Wedding']
    for event_name in event_names:
        event_dir = os.path.join(training_data_dir, event_name)
        os.makedirs(event_dir, exist_ok=True)
        for i in range(5):  # Create 5 dummy images per event
            image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            cv2.imwrite(os.path.join(event_dir, f'{i}.jpg'), image)

def test_organize_photos(setup_input_output_dirs):
    input_dir, output_dir = setup_input_output_dirs
    organize_photos(input_dir, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'test.jpg'))

def test_train_model():
    with tempfile.TemporaryDirectory() as training_data_dir:
        create_dummy_data(training_data_dir)
        train_model(training_data_dir)
        
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
