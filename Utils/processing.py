import pandas as pd
import numpy as np
import cv2

from Utils.constants import AUGMENTATION_FUNCTIONS

# Image Augmentation
def augment_data(data, augmentation_functions= AUGMENTATION_FUNCTIONS):
    """This function applies various augmentation techniques to the input data.

    Args:
        data (dict): A dictionary containing 'image' and 'Name' keys. 
                     'image' is a list of images (np.array) and 'Name' is a list of corresponding labels.
                     
        augmentation_functions (list of func objects, optional): A list of augmentation functions to be applied to the images.

    Returns:
        pd.DataFrame: A DataFrame containing augmented images, their corresponding labels, 
                      and the name of the augmentation function applied.
    """
    augmented_images = []
    augmented_labels = []
    augmented_fun = []
    for func in augmentation_functions:
        print(f"Applying {func.__name__} to images")
        for img, label in zip(data['image'], data['name']):
            augmented_img = func(img)
            augmented_images.extend(augmented_img)
            augmented_labels.extend([label]*len(augmented_img))
            augmented_fun.extend([func.__name__]*len(augmented_img))
        print(f"Done applying {func.__name__} to images")
    
    augmented_data = pd.DataFrame({'image': augmented_images,
                                   'name': augmented_labels, 
                                   'function': augmented_fun})
    return augmented_data


# preprocessing
def preprocess_image(img, size=(128, 128)):
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # remove gaussian noiss
    img = cv2.GaussianBlur(img, (5, 5), 0) 
    
    # # remove salt and pepper noise
    # img = cv2.medianBlur(img, 5)
    
    # shaprening to enhance the edges and solve bluriness
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    
    # equalize the histogram to enhance the contrast
    img = cv2.equalizeHist(img)
    
    img = img / 255.0
    return img


