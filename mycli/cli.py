import click
from mycli.commands.photo_organizer import PhotoOrganizer

@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_dir')
@click.argument('output_dir')
def organize(input_dir, output_dir):
    """Organize photos in the input directory and save to the output directory."""
    organizer = PhotoOrganizer()
    organizer.organize_photos(input_dir, output_dir)

@cli.command()
@click.argument('training_data_dir')
def train(training_data_dir):
    """Train the machine learning model with the provided training data."""
    organizer = PhotoOrganizer()
    organizer.train_model(training_data_dir)

@cli.command()
@click.argument('input_dir')
@click.argument('output_dir')
def classify(input_dir, output_dir):
    """Classify and organize photos based on the trained model."""
    organizer = PhotoOrganizer()
    organizer.classify_and_organize_photos(input_dir, output_dir)

@cli.command()
@click.argument('image_path')
def classify_single(image_path):
    """Classify a single photo and print the predicted label."""
    organizer = PhotoOrganizer()
    predicted_label = organizer.classify_photo(image_path)
    print(f"The predicted label for the image is: {predicted_label}")

if __name__ == '__main__':
    cli()
