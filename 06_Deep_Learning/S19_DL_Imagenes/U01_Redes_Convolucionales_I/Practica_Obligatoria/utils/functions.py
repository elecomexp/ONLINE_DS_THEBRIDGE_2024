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


def load_images_from_directory(directory, reshape_dim=(32, 32)):
    """
    Loads images from the specified directory, resizes them to the given dimensions,
    and extracts the class labels from the file names using the prefix before the dot.

    Parameters
    ----------
    directory : str
        Path to the directory containing the images.
    reshape_dim : tuple
        The target shape for resizing the images (width, height).

    Returns
    -------
    X (np.array)
        Array of images resized to `reshape_dim`.
    y (np.array)
        Array of labels extracted from the image file names.
    
    Example
    -------
    >>> train_directories = [
    >>>     'data/github_train_0',
    >>>     'data/github_train_1'
    >>> ]

    >>> X_train, y_train = [], []
    >>> for train_dir in train_directories:
    >>>     X, y = load_images_from_directory(train_dir, reshape_dim=(32, 32))
    >>>     X_train.extend(X)
    >>>     y_train.extend(y)

    >>> X_train = np.array(X_train)
    >>> y_train = np.array(y_train)
    """
    X = []
    y = []
    
    # Loop through each file in the directory
    for file in os.listdir(directory):
        if file.endswith('.jpg'):  # Only process jpg files
            # Read the image
            image = imread(os.path.join(directory, file))
            # Resize the image to the specified dimensions
            image = cv2.resize(image, reshape_dim)
            # Append the image to the list
            X.append(image)
            # Extract the label based on the file name before the dot
            label = file.split('.')[0]
            y.append(label)

    return np.array(X), np.array(y)


def show_images_batch(images, titles=[], n_cols=5, size_scale=2, cmap="Greys"):
    """
    Displays a batch of images in a grid layout using matplotlib.

    Parameters:
    -----------
    images : list or array-like
        A list or array containing image data to be displayed. Each element
        should be an individual image (2D or 3D array depending on the image type).
    
    titles : list, optional (default=[])
        A list of titles for each image. If provided, each image will be displayed
        with its corresponding title. The length of this list should match the
        number of images. If no titles are provided, no titles will be shown.
    
    n_cols : int, optional (default=5)
        The number of columns in the grid. This determines how many images will
        be displayed per row. The total number of rows will be computed automatically.
    
    size_scale : float, optional (default=2)
        Scaling factor for the size of the images. Larger values will increase the
        size of each individual image in the grid.
    
    cmap : str, optional (default='Greys')
        The color map to use when displaying the images. Useful when showing grayscale
        images. You can change it to other colormaps like 'viridis', 'plasma', etc.
    
    Returns:
    --------
    None
        This function displays a plot of the images but does not return any values.

    Examples:
    ---------
    >>> show_images_batch([image1, image2, image3], titles=['Image 1', 'Image 2', 'Image 3'], n_cols=3)
    """
    
    n_rows = (len(images) - 1) // n_cols + 1
    plt.figure(figsize=(n_cols * size_scale, n_rows * 1.1 * size_scale))
    
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap=cmap)
        plt.axis("off")
        if len(titles):
            plt.title(titles[index])
    plt.tight_layout()
    plt.show()
    

def check_image_data(sets):
    """
    Checks for missing or non-finite values in image datasets.

    Parameters
    ----------
    sets : List of NumPy arrays containing image data.

    Prints
    ------
    Status of each dataset: whether it has missing values, non-finite values, or is OK.
    
    Example
    -------
    >>> check_image_data([X_train, y_train, X_test, y_test])
    """
    for set_num, data_set in enumerate(sets):
        if np.isnan(data_set).any():
            print(f'There are missing values in "{set_num}" image data.')
        elif not np.isfinite(data_set).all():
            print(f'There are non-finite values (Inf) in "{set_num}" image data.')
        else:
            print(f'Set {set_num}: OK.')


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


def select_misclassified_images(X_test, y_test, y_pred, y_pred_proba, top_percentage=0.1):
    """
    Selects the top percentage of misclassified images with the highest confidence in the wrong prediction.
    
    Parameters:
    -----------
    X_test : array-like
        The test set of images.
    
    y_test : array-like
        The true labels of the test set.
    
    y_pred : array-like
        The predicted labels of the test set.
    
    y_pred_proba : array-like
        The predicted probabilities of the test set for each class.
    
    top_percentage : float, optional (default=0.1)
        The top percentage of misclassified images to select based on confidence in the wrong class.
    
    Returns:
    --------
    top_misclassified_images : list
        The list of images corresponding to the top misclassified cases.
    
    top_misclassified_titles : list
        The list of titles for the misclassified images showing true and predicted labels along with the confidence.
    """
    # Compute the maximum confidence for each prediction
    confidence = [prediction.max() for prediction in y_pred_proba]

    # Create a DataFrame with true labels, predicted labels, and confidence
    pred_df = pd.DataFrame({
        "True": y_test,
        "Predicted": y_pred,
        "Confidence": confidence
    })

    # Identify misclassified samples
    errors = pred_df["True"] != pred_df["Predicted"]
    misclassified_df = pred_df[errors].copy()

    # Sort the misclassified samples by confidence in descending order
    misclassified_df = misclassified_df.sort_values("Confidence", ascending=False)

    # Select the top percentage of misclassified images
    top_n = int(len(misclassified_df) * top_percentage)
    top_misclassified = misclassified_df.head(top_n)

    # Extract the indices of the top misclassified images
    top_misclassified_indices = top_misclassified.index.tolist()

    # Extract the corresponding images and titles for display
    top_misclassified_images = X_test[top_misclassified_indices]
    top_misclassified_titles = [
        f"True: {true_label}, Pred: {pred_label} (Conf: {conf:.2f})"
        for true_label, pred_label, conf in zip(top_misclassified["True"], top_misclassified["Predicted"], top_misclassified["Confidence"])
    ]

    return top_misclassified_images, top_misclassified_titles
