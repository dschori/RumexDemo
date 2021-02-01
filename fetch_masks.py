import argparse
import json
import os
import urllib.request
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='Fetch ground truth masks from .json File')

parser.add_argument('json_path',
                    type=str,
                    help='The Json File path')
parser.add_argument('output_image_path',
                    type=str,
                    help='The image storage path')
parser.add_argument('output_mask_path',
                    type=str,
                    help='The mask storage path')

args = parser.parse_args()

Path(args.output_image_path).mkdir(parents=True, exist_ok=True)


def get_image(download_link, is_grey):
    base64_data = urllib.request.urlopen(download_link).read()
    pil_image = Image.open(BytesIO(base64_data))
    if is_grey:
        pil_image = pil_image.convert('L')
    np_image = np.array(pil_image).astype(np.uint8)
    return np_image


def get_mask(label_name, image_object, resolution):
    msk = np.zeros(resolution, dtype=np.uint8)
    if len(image_object) == 0:
        return msk
    else:
        try:
            for object in image_object['Label']['objects']:
                if object['value'] == label_name:
                    msk += get_image(download_link=object['instanceURI'],
                                     is_grey=True)
            return np.clip(msk, 0, 255)
        except:
            return msk


def save_image(path, img):
    plt.imsave(path, img)


labels_to_store = ['rumex-leaf', 'garbage']

with open(args.json_path) as f:
    label_data = json.load(f)
    for i, image in enumerate(label_data):
        if len(image['Label']) == 0:
            continue
        image_file = get_image(download_link=image['Labeled Data'],
                               is_grey=False)
        image_name, image_ext = os.path.splitext(image['External ID'])
        save_image(path='{}/{}{}'.format(args.output_image_path, image_name, image_ext), img=image_file)

        msks = np.zeros((1200, 5600, 3), dtype=np.uint8)
        for i, label_name in enumerate(labels_to_store):
            msk = get_mask(label_name=label_name, image_object=image, resolution=(1200, 5600))
            msks[:, :, i] = msk
        background = np.clip(msks[:, :, 0] + msks[:, :, 1], 0, 1) - 1*-255
        indices = background == 256
        background[indices] = 0
        msks[:, :, 2] = background

        Path('{}/{}'.format(args.output_mask_path, label_name)).mkdir(parents=True, exist_ok=True)
        save_image(path='{}/{}/{}{}'.format(args.output_mask_path, label_name, image_name, image_ext), img=msks)


print('Finished, stored Masks in: {}'.format(args.output_mask_path))
