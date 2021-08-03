from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]

setup(
    name="help_ML",
    version="0.0.1",
    author="Danilo Pereira",
    author_email="danpsbr@gmail.com",
    description="A package to Machine Learning Notebook",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/danps/Machine_Learning",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache License",
    ],
)