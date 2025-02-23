import click
from mycli.commands.photo_organizer import organize_photos, train_model

@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_dir')
@click.argument('output_dir')
def organize(input_dir, output_dir):
    """Organize photos in the input directory and save to the output directory."""
    organize_photos(input_dir, output_dir)

@cli.command()
@click.argument('training_data_dir')
def train(training_data_dir):
    """Train the machine learning model with the provided training data."""
    train_model(training_data_dir)

if __name__ == '__main__':
    cli()
