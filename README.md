# mycli
CLI for completing regular tasks

## Overview

`mycli` is a command-line interface (CLI) tool for organizing photos using machine learning. It can classify events and people in photos, and rename the photos with a date, people's names, and events. The tool also allows you to train the machine learning model through the CLI.

## Installation

To install the `mycli` package, run the following command:

```bash
pip install .
```

## Requirements

Make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Organize Photos

To organize photos in a directory, use the `organize` command. This command takes two arguments: the input directory containing the photos and the output directory where the organized photos will be saved.

```bash
mycli organize <input_dir> <output_dir>
```

Example:

```bash
mycli organize /path/to/input_photos /path/to/output_photos
```

### Train Model

To train the machine learning model with the provided training data, use the `train` command. This command takes one argument: the directory containing the training data.

```bash
mycli train <training_data_dir>
```

Example:

```bash
mycli train /path/to/training_data
```

## Development

To contribute to the development of `mycli`, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mycli.git
    ```

2. Navigate to the project directory:
    ```bash
    cd mycli
    ```

3. Install the package in editable mode along with the development dependencies:
    ```bash
    pip install -e .[dev]
    ```

4. Run the tests to ensure everything is working:
    ```bash
    python -m unittest discover tests
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
