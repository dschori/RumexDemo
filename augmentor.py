import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'data/images'
mask_path = 'data/masks'

images = ['img_1001.png',
          'img_1002.png',
          'img_1003.png',
          'img_1004.png']

masks = ['img_1000.png']

# (upper left, lower right)
# [(x1, y1), (x2, y2)]
mask_ranges = [
    [(1400, 340), (2020, 930)],
    [(2500, 440), (3240, 1090)],
    [(3230, 130), (3760, 720)],
    [(4370, 470), (5050, 1030)]
]

for i in range(500):
    img_ind = np.random.randint(0, len(images))
    img = plt.imread('{}/{}'.format(image_path, images[img_ind]))
    img = (img*255.).astype(np.uint8)
    #img = cv2.imread('{}/{}'.format(image_path, images[img_ind]), 3)


    img1 = img.copy()
    new_mask = np.zeros_like(img1)
    for _ in range(np.random.randint(0, 6)):

        msk_ind = np.random.randint(0, len(masks))
        msk = cv2.imread('{}/{}'.format(mask_path, masks[msk_ind]))
        #msk = plt.imread('{}/{}'.format(mask_path, masks[msk_ind]))
        #msk = (msk*255.).astype(np.uint8)

        crop_ind = np.random.randint(0, len(mask_ranges))

        #rumex_img = cv2.imread('{}/{}'.format(image_path, masks[msk_ind]))
        rumex_img = plt.imread('{}/{}'.format(image_path, masks[msk_ind]))
        rumex_img = (rumex_img*255.).astype(np.uint8)

        #foreground = rumex_img * msk
        foreground = cv2.bitwise_and(rumex_img, msk)

        cropped_foreground = foreground[mask_ranges[crop_ind][0][1]:mask_ranges[crop_ind][1][1],
                      mask_ranges[crop_ind][0][0]:mask_ranges[crop_ind][1][0],
                      :]

        cropped_mask = msk[mask_ranges[crop_ind][0][1]:mask_ranges[crop_ind][1][1],
                      mask_ranges[crop_ind][0][0]:mask_ranges[crop_ind][1][0],
                      :]

        if np.random.rand() > 0.5:
            cropped_foreground = cropped_foreground[:, ::-1]
            cropped_mask = cropped_mask[:, ::-1]
        if np.random.rand() > 0.5:
            cropped_foreground = cropped_foreground[::-1, :]
            cropped_mask = cropped_mask[::-1, :]

        fx = np.random.uniform(0.7,1.3)
        fy = np.random.uniform(0.7,1.3)
        cropped_foreground = cv2.resize(cropped_foreground, None, fx=fx, fy=fy)
        cropped_mask = cv2.resize(cropped_mask, None, fx=fx, fy=fy)

        img2 = cropped_foreground.copy()
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # I want to put logo on top-left corner, So I create a ROI
        rows, cols, channels = img2.shape
        row_offset = np.random.randint(0, img1.shape[0]-img2.shape[0])
        col_offset = np.random.randint(0, img1.shape[1]-img2.shape[1])
        min_row, max_row = row_offset, row_offset+img2.shape[0]
        min_col, max_col = col_offset, col_offset+img2.shape[1]
        roi = img1[min_row:max_row, min_col:max_col]
        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        img1[min_row:max_row, min_col:max_col] = dst
        new_mask[min_row:max_row, min_col:max_col] += cropped_mask

    #plt.imsave('data/images_aug/img_{}.png'.format(i+1100), img1)
    cv2.imwrite('data/images_aug/img_{}.png'.format(i+1100), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    new_mask = np.clip(new_mask, 0, 255)
    cv2.imwrite('data/masks_aug/img_{}.png'.format(i+1100), cv2.cvtColor(new_mask, cv2.COLOR_RGB2BGR))
