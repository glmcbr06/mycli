from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mycli',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mycli=mycli.cli:cli',
        ],
    },
)
