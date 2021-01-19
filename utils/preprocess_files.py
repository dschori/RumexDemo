import cv2
import os

raw_images_folder = '../data/raw'
save_images_folder = '../data/images'

img_prefix = 'img_'

img_offset = 1000

ORI_HEIGTH, ORI_WIDTH = 2000, 7000

for ind, image_file in enumerate(os.listdir(raw_images_folder)):
    img = cv2.imread('{}/{}'.format(raw_images_folder, image_file))
    img = cv2.resize(img, dsize=(ORI_WIDTH, ORI_HEIGTH))
    img = img[400:-400, 700:-700]
    cv2.imwrite('{}/{}{}.png'.format(save_images_folder, img_prefix, ind+img_offset), img)
    print(img.shape)
    
print('Finished, Processed {} Files'.format(ind+1))