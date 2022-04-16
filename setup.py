
import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="BTtools",
    version="V6.1",
    author="Symen Hovinga",
    author_email="itsfull@hotmail.com",
    description="Tools for processing Burrtools xmpuzzle files in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Seemee/BTtools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["lxml","pathLib","numpy"]
)
