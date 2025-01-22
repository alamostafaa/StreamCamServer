import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os

from Utils.constants import DIR_PATH

def read_images(dir_path=DIR_PATH):
    """This function reads images from the given directory path and returns a DataFrame containing images and their corresponding labels.

    Args:
        dir_path (str, optional): The path to the directory containing images. Defaults to DIR_PATH.
    Returns:
        data (dataframe): A DataFrame containing images and their corresponding labels.
    """
    images = []
    labesl = []
    for file in os.listdir(dir_path):
        for image in os.listdir(f'{dir_path}/{file}'):
            img_path = f'{dir_path}/{file}/{image}'
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labesl.append(file) 
            else:
                print(f"Failed to read image: {img_path}")

    data = pd.DataFrame({'image': images, 'name': labesl})
    print(data.head())
    return data

def save_images(data, dir_path=DIR_PATH):
    """This function saves images to the given directory path.
    
        Args: 
            data (dataframe): A DataFrame containing images and their corresponding labels.
            dir_path (str, optional): The path to the directory where images will be saved. Defaults to DIR_PATH.
            
        Returns: 
            None        
    """
    for i, row in data.iterrows():
        img = row['image']
        name = row['name']
        func = row['function'] if 'function' in row else ''

        img_path = f'{dir_path}/{name}/{func}_{i}.jpg'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, img)
        print(f"Saved image to {img_path}")

# Display augmented images
def dispay_image(data, based_on):
    """This function displays augmented images based on the given label.
    
    Args:
        data (dataframe): A DataFrame containing images and their corresponding labels.
        based_on (str): The label based on which images will be displayed.  
        
    Returns:
        None
    """
    based_labels = data[f'{based_on}'].unique()
    fig, axes = plt.subplots(nrows=len(based_labels), ncols=5, figsize=(28, 28))
    for i, txt in enumerate(based_labels):
        func_images = data[data[f'{based_on}'].str.contains(txt)]
        if (len(func_images) == 0):
            print(f"No images found for {txt}")
            continue
        random_images = func_images.sample(n=5, random_state=42)
        for ax, img, name in zip(axes[i], random_images['image'], random_images[f'{based_on}']):
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(name)
    plt.show()
