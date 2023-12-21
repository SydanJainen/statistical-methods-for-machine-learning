from setuptools import setup, find_packages

setup(
    name='spotify-ridge-regression',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
    ],
)