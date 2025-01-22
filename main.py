from Utils.image_helpers import read_images, dispay_image
from Utils.processing import preprocess_image


print('Reading images...')
data = read_images()
print('Reading images completed\n')
print('Preprocessing images...')
data['image'] = data['image'].apply(preprocess_image)
print('Preprocessing images completed\n')
print('Displaying images...')
dispay_image(data, 'name')
print('Displaying images completed\n')
