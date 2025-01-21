from components.augmentation_functions import motion_blur, gaussian_noise, contrast, salt_pepper_noise, \
    scale, rotate, flip
    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
import random

def read_images(dir_path='OUR dataset'):
    images = []
    labesl = []
    for file in os.listdir(dir_path):
        for image in os.listdir(f'{dir_path}/{file}'):
            img_path = f'{dir_path}/{file}/{image}'
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for displaying with matplotlib
                images.append(img)
                labesl.append(file) 
            else:
                print(f"Failed to read image: {img_path}")
    
    # Display images in a grid
    # fig, axes = plt.subplots(nrows=len(images)//5 + 1, ncols=5, figsize=(15, 15))
    # for ax, img in zip(axes.flat, images):
    #     ax.imshow(img)
    #     ax.axis('off')
    # plt.show()
    data = pd.DataFrame({'image': images, 'Name': labesl})
    print(data.head())
    return data

data = read_images()

# Apply augmentation functions
augmentation_functions = [motion_blur, gaussian_noise, salt_pepper_noise, contrast, scale, rotate, flip]

def augment_data(data, augmentation_functions):
    augmented_images = []
    augmented_labels = []
    augmented_fun = []
    for func in augmentation_functions:
        print(f"Applying {func.__name__} to images")
        for img, label in zip(data['image'], data['Name']):
            augmented_img = func(img)
            augmented_images.extend(augmented_img)
            augmented_labels.extend([label]*len(augmented_img))
            augmented_fun.extend([func.__name__]*len(augmented_img))
        print(f"Done applying {func.__name__} to images")
    
    augmented_data = pd.DataFrame({'image': augmented_images, 'Name': augmented_labels, 'Function': augmented_fun})
    return augmented_data

augmented_data = augment_data(data, augmentation_functions)
# print(augmented_data.head(15))

# Display augmented images
# def dispay_image(augmentation_functions, augmented_data):
#     fig, axes = plt.subplots(nrows=len(augmentation_functions), ncols=5, figsize=(40, 40))
#     for i, func in enumerate(augmentation_functions):
#         func_name = func.__name__
#         func_images = augmented_data[augmented_data['Function'].str.contains(func_name)]
#         if (len(func_images) == 0):
#             print(f"No images found for {func_name}")
#             continue
#         random_images = func_images.sample(n=5, random_state=42)
#         for ax, img, name in zip(axes[i], random_images['image'], random_images['Name']):
#             ax.imshow(img)
#             ax.axis('off')
#             ax.set_title(name)

# dispay_image(augmentation_functions, augmented_data)
# plt.show()

# print(f"{len(augmented_data)} and before it was {len(data)}")

def save_images(augmented_data, dir_path='OUR dataset'):
    for i, row in augmented_data.iterrows():
        img = row['image']
        name = row['Name']
        func = row['Function']

        img_path = f'{dir_path}/{name}/{func}_{i}.jpg'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, img)
        print(f"Saved image to {img_path}")
save_images(augmented_data)