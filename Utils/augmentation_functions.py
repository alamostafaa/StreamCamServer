import numpy as np
import cv2

from Utils.constants import AUGMENTATION_FUNCTIONS


def motion_blur(img, size=30, angle=45):
    """this function adds motion blur noise to image to camera shake effect

    Args:
        img (np.array): input image
        size (int, optional): the size of the kernal. Defaults to 15.
        angle (int, optional): the angel of the shake (applied motion blur). Defaults to 45.
    """
    kernal = np.zeros((size, size))
    kernal[size//2, :] = np.ones(size)
    kernal = cv2.warpAffine(kernal, cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0), (size, size))
    kernal = kernal/size # normalize the kernal
    blurred_img = cv2.filter2D(img, -1, kernal)
    
    return [blurred_img]


def gaussian_noise(img, mean=0, std=35):
    """this function adds gaussian noise to the image to make it grainy

    Args:
        img (np.array): input image
        mean (int, optional): mean of the noise. Defaults to 0.
        std (float, optional): standard deviation of the noise. Defaults to 25.
    """
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy_image = cv2.add(img.astype(np.float32), noise)
    return [np.clip(noisy_image, 0, 255).astype(np.uint8)]


def salt_pepper_noise(img, amount=0.02):
    """this function adds salt and pepper noise to the image

    Args:
        img (np.array): input image
        amount (float, optional): the amount of noise to be added. Defaults to 0.01.
    """
    noisy_img = img.copy()
    num_salt = np.ceil(amount * img.size * 0.5)
    num_pepper = np.ceil(amount * img.size * 0.5)
    
    # add salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[coords[0], coords[1], :] = 1
    
    # add pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[coords[0], coords[1], :] = 0
    
    return [noisy_img]


def contrast(img):
    """this function adds contrast to the image by scaling the pixel values

    Args:
        img (np.array): input image
        alpha (float, optional): contrast factor. Defaults to 1.5.
        beta (int, optional): brightness factor. Defaults to 0.
    """
    high_contrast_img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    low_contrast_img = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
    
    bright_img = cv2.convertScaleAbs(img, alpha=1, beta=50)
    dark_img = cv2.convertScaleAbs(img, alpha=1, beta=-50)
    
    return [high_contrast_img, low_contrast_img, bright_img, dark_img]



def scale(img, scale=2):
    """this function simulates zoom in and zoom out by scaling the image

    Args:
        img (np.array): input image
        scale (int, optional): scale factor. Defaults to 2.
    """
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Zoom in
    zoom_in_img = cv2.resize(img, (width * scale, height * scale))
    zoom_in_img = zoom_in_img[center_y:center_y + height, center_x:center_x + width]

    # Zoom out
    zoom_out_img = cv2.resize(img, (width // scale, height // scale))
    zoom_out_img = cv2.copyMakeBorder(zoom_out_img, 
                                      (height - zoom_out_img.shape[0]) // 2, 
                                      (height - zoom_out_img.shape[0]) // 2, 
                                      (width - zoom_out_img.shape[1]) // 2, 
                                      (width - zoom_out_img.shape[1]) // 2, 
                                      cv2.BORDER_REPLICATE)

    return [zoom_in_img, zoom_out_img]


def rotate(img):
    """this function rotates the image by angle degrees

    Args:
        img (np.array): input image
        angle (int, optional): angle of rotation. Defaults to 45.
    """
    angles =[45, 90, 135, 180, 225, 270, 315]
    images = []
    for angle in angles:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        images.append(rotated_img)
    
    return images

def flip(img, flip_code=1):
    """this function flips the image

    Args:
        img (np.array): input image
        flip_code (int, optional): flip code. Defaults to 1.
    """
    flipped_img = cv2.flip(img, flip_code)
    
    return [flipped_img]
