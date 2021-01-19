import argparse
import os
import warnings

import numpy as np
from sklearn.model_selection import train_test_split

from utils.dataset_utils import get_dataset
from utils.unet_utils import Train

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='U-Net Trainer')

# parser.add_argument('train_path',
#                    type=str,
#                    help='Path to Training Images/Masks Pairs',
#                    required=False)

# args = parser.parse_args()

image_dir = 'data/imgs/'
mask_dir = 'data/msks/'


def create_subsets(image_list, mask_list, train_size=0.8):
    ind = np.arange(len(image_list))
    train_ind, val_ind = train_test_split(ind, train_size=train_size)
    train_image_list = [image_list[ind] for ind in train_ind]
    train_mask_list = [mask_list[ind] for ind in train_ind]
    val_image_list = [image_list[ind] for ind in val_ind]
    val_mask_list = [mask_list[ind] for ind in val_ind]
    return train_image_list, train_mask_list, val_image_list, val_mask_list


image_list = ['{}{}'.format(image_dir, image_name) for image_name in os.listdir(image_dir)]
mask_list = ['{}{}'.format(mask_dir, mask_name) for mask_name in os.listdir(mask_dir)]

assert len(image_list) == len(mask_list), 'Number of Images and Masks are not equal'

train_image_list, train_mask_list, val_image_list, val_mask_list = \
    create_subsets(image_list=image_list, mask_list=mask_list, train_size=0.8)

train_set = get_dataset(image_list=train_image_list,
                        mask_list=train_mask_list,
                        do_augmentations=True)

val_set = get_dataset(image_list=val_image_list,
                      mask_list=val_mask_list,
                      do_augmentations=False)

output = list(train_set.take(1))[0]
sample_img, sample_msk = output[0], output[1]

print('Got images_old with shape: {}'.format(sample_img.shape))
print('Got masks with shape: {}'.format(sample_msk.shape))

trainer = Train(train_set=train_set, val_set=val_set)

trainer.create_model(output_channels=1, backbone_name='resnet50', backbone_trainable=True)

print(trainer.model.summary())

trainer.fit(train_set_size=len(train_image_list))
