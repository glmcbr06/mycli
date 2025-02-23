from setuptools import setup, find_packages

setup(
    name='mycli',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'click',
        'scikit-learn',  # Assuming scikit-learn for ML tasks
        'opencv-python',  # Assuming OpenCV for image processing
    ],
    entry_points={
        'console_scripts': [
            'mycli=mycli.cli:cli',
        ],
    },
)
