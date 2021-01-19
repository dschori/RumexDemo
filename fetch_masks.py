import os
import json
import argparse
import urllib.request
from pathlib import Path

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
        download_link = image['Label']['objects'][0]['instanceURI']
        image_name, image_ext = os.path.splitext(image['External ID'])
        print('Fetching Image {} of {}: {}'.format(i + 1,
                                                   len(label_data),
                                                   image_name + image_ext))
        img = urllib.request.urlretrieve(download_link, "{}/{}.jpg".format(args.output_path,
                                                                           image_name))

print('Finished, stored Masks in: {}'.format(args.output_path))
