# make sure that the exported JSON file and this script are in the same directory
# replace filename with appropriate
from PIL import Image
import requests, urllib, json, os, csv, sys
from urllib.request import urlopen


# for extracting masks from JSON exports
def parseJSON(filename):
    with open(filename) as file:
        data = file.read()
        print(type(data))
        # we use json loads to be able to parse data
        decoded_data = json.loads(data)
        print(type(decoded_data))
        # res := list containing all instanceURIs
        res = []
        num_labels = len(decoded_data)
        for i in range(num_labels):
            object_data = decoded_data[i]['Label']['objects']
            print(type(decoded_data[i]))
            print(decoded_data[i])

            for j in range(len(object_data)):
                res.append(object_data[j]['instanceURI'])

        return res


# for extracting masks from CSV exports
def parseCSV(filename):
    mask_list = []

    with open(filename, 'r') as file:
        readCSV = csv.reader(file, delimiter=',')
        for row in readCSV:
            # mask URLs are stored in column 17
            tokenized = row[17].split(":")
            if len(tokenized) == 3:
                mask_URL = tokenized[1] + ":" + tokenized[2][:-1]
                mask_list.append(mask_URL[1:-1])
    return mask_list


# opens all exported data images. use with caution
def readAllImageData(image_data):
    for url in image_data:
        response = requests.get(url)
        img = Image.open(urlopen(url))
        img.show()


def downloadData(url_list):
    # creating a LabelboxData folder in user's local machine's download directory
    download_path = os.path.expanduser("~") + "/Downloads/LabelboxData/"
    print(url_list)
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    print("Beginning download for Labelbox exported data...")

    for i in range(len(url_list)):
        # generating unique image names
        # Format: LabelID_Class_mask1.jpg
        file_name = download_path + url_list[i][0] + "_" + url_list[i][1] + "_mask" + str(i) + ".png"
        print(file_name)
        urllib.request.urlretrieve(url_list[i], file_name)


if __name__ == '__main__':
    # to run: python3 download_masks.py <FILENAME>

    exported_data = parseJSON('Set1.json')
    # exported_data = parseCSV(sys.argv[1])
    downloadData(exported_data)
