import os
import shutil
import unittest
from mycli.commands.photo_organizer import organize_photos, train_model

class TestPhotoOrganizer(unittest.TestCase):

    def setUp(self):
        self.input_dir = 'test_input'
        self.output_dir = 'test_output'
        os.makedirs(self.input_dir, exist_ok=True)
        with open(os.path.join(self.input_dir, 'test.jpg'), 'w') as f:
            f.write('test image content')

    def tearDown(self):
        shutil.rmtree(self.input_dir)
        shutil.rmtree(self.output_dir)

    def test_organize_photos(self):
        organize_photos(self.input_dir, self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'test.jpg')))

    def test_train_model(self):
        train_model(self.input_dir)  # Just a placeholder test

if __name__ == '__main__':
    unittest.main()
