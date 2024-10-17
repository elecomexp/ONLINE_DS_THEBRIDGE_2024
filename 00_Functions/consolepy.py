import os
import shutil
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.io import imread

def unzip_files(zip_file_path, extract_to_dir):
    """
    Unzips a zip file to the specified directory.
    
    Parameters
    ----------
    zip_file_path : str
        Path to the zip file.
    extract_to_dir : str
        Directory where the contents of the zip file will be extracted.
        
    Example
    -------
    >>> zip_files = [
    >>>     './data/github_train_0.zip',
    >>>     './data/github_train_1.zip',
    >>>     './data/github_train_2.zip',
    >>>     './data/github_train_3.zip',
    >>>     './data/github_test.zip'
    >>> ]

    >>> for zip_file in zip_files:
    >>>     unzip_files(zip_file, './data/')
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
    print(f"Extracted {zip_file_path} to {extract_to_dir}")
    

def remove_non_zip_folders(base_path):
    """
    Recursively removes all folders that do not contain any .zip files.

    Parameters
    ----------
        base_path (str): The root directory to start the search from.

    Example
    -------
        If you have the following folder structure:
        
        base_directory/
        ├── folder1/
        │   └── file1.txt
        ├── folder2/
        │   └── archive.zip
        └── folder3/
            └── image.png

        Calling remove_non_zip_folders(base_directory) will remove 'folder1' and 'folder3',
        but will keep 'folder2' because it contains a .zip file.
    
    Usage:
        base_directory = "path_to_your_folder/data"
        remove_non_zip_folders(base_directory)
    """
    # Recursively walk through all folders and subfolders
    for root, dirs, files in os.walk(base_path, topdown=False):
        # Check if the current folder contains any .zip file
        has_zip = any(file.endswith('.zip') for file in files)
        
        # If no .zip files are found, remove the folder
        if not has_zip:
            print(f"Removing folder: {root}")
            shutil.rmtree(root)


def remove_npy_files(base_path):
    """
    Recursively removes all .npy files from the specified directory and its subdirectories.

    Args:
        base_path (str): The root directory to start the search from.

    Example:
        If you have the following folder structure:
        
        base_directory/
        ├── folder1/
        │   └── data.npy
        ├── folder2/
        │   └── script.py
        └── file.npy

        Calling remove_npy_files(base_directory) will remove 'data.npy' and 'file.npy',
        but will leave 'script.py' untouched.

    Usage:
        base_directory = "path_to_your_folder/data"
        remove_npy_files(base_directory)
    """
    # Recursively walk through all folders and subfolders
    for root, dirs, files in os.walk(base_path):
        # Remove each .npy file found
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                print(f"Removing file: {file_path}")
                os.remove(file_path)