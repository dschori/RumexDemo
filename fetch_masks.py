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
parser.add_argument('output_path',
                    type=str,
                    help='The mask storage path')

args = parser.parse_args()

Path(args.output_path).mkdir(parents=True, exist_ok=True)

download_links = []
with open(args.json_path) as f:
    label_data = json.load(f)
    for i, image in enumerate(label_data):
        image_name, image_ext = os.path.splitext(image['External ID'])
        mask_name = "{}/{}.png".format(args.output_path,
                                       image_name)
        objects = image['Label']['objects']
        if len(objects) == 0:
            print('Storing empty mask: {}'.format(mask_name))
            plt.imsave(mask_name, np.zeros((1200, 5600), dtype=np.uint8), cmap='gray')
        else:
            print('Fetching Image {} of {}: {}'.format(i + 1,
                                                       len(label_data),
                                                       image_name + image_ext))
            msk = np.zeros((1200, 5600), dtype=np.uint8)
            for object in objects:
                if object['value'] == 'rumex-leaf':
                    download_link = object['instanceURI']
                    data = urllib.request.urlopen(download_link).read()
                    #r_data = binascii.unhexlify(data)
                    pil_image = Image.open(BytesIO(data)).convert('L')
                    tmp_msk = np.array(pil_image).astype(np.uint8)
                    msk += tmp_msk
            msk = np.clip(msk, 0, 255)
            plt.imsave(mask_name, msk, cmap='gray')
            #plt.imshow(msk)
            #plt.show()


print('Finished, stored Masks in: {}'.format(args.output_path))
