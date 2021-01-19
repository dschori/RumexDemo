import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import Config


class ImageViewer():
    def __init__(self):
        pass

    def get_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError('Image with Path: "{}" not found'.format(path))
        return img

    def get_mask(self, path):
        msk = cv2.imread(path)
        if msk is None:
            raise ValueError('Image with Path: "{}" not found'.format(path))
        return msk

    def get_pair(self, index):
        img_path = '{}/images/img_{}.png'.format(Config.folder_path, index)
        msk_path = '{}/masks/img_{}.png'.format(Config.folder_path, index)
        img, msk = self.get_image(img_path), self.get_mask(msk_path)
        return img, msk

    def get_overlaid_pair(self, index, alpha=0.4):
        img, msk = self.get_pair(index)
        img = img.astype(dtype=np.float) / 255.
        msk = msk.astype(dtype=np.float) / 255.

        msk *= alpha

        # make overlay
        color = (1, 0, 0.4)
        overlay = np.ones(img.shape, dtype=np.float) * color

        # overlay over original image
        out = overlay * msk + img * (1.0 - msk)
        return out

    def show_pair(self, index, alpha=0.4):
        overlay = self.get_overlaid_pair(index=index, alpha=alpha)
        plt.imshow(overlay)
        plt.show()


def get_dataset(image_list, mask_list, do_augmentations=False):
    """

    :param image_list:
    :param mask_list:
    :param buffer_size:
    :param batch_size:
    :return:
    """
    dataset_images = tf.data.Dataset.list_files(image_list, shuffle=False)
    dataset_masks = tf.data.Dataset.list_files(mask_list, shuffle=False)
    dataset = tf.data.Dataset.zip((dataset_images, dataset_masks))

    for func in [process_path, random_crop]:
        dataset = dataset.map(func, num_parallel_calls=Config.tf_parallel_calls)

    if do_augmentations:
        for func in [random_brightness, random_flip, add_gaussian_noise, write_file]:
            dataset = dataset.map(func, num_parallel_calls=Config.tf_parallel_calls)

    for func in [mask_to_grayscale]:
        dataset = dataset.map(func, num_parallel_calls=Config.tf_parallel_calls)

    dataset = dataset.batch(Config.batch_size)
    dataset = dataset.shuffle(Config.shuffle_size, reshuffle_each_iteration=True)
    return dataset


def predict_image():
    pass


def slice_image():
    pass


def write_file(image, mask, image_path):
    #img = tf.io.encode_jpeg(image=tf.image.convert_image_dtype(image, tf.uint8), format='rgb')
    #tf.io.write_file('test.png', img)

    tmp_mask = tf.math.multiply(mask, tf.constant(0.4))

    # make overlay
    #color = (1, 0, 0.4)
    overlay = tf.math.multiply(tf.ones_like(image), tf.constant([1., 0., 0.4]))
    #overlay = np.ones(img.shape, dtype=np.float) * color

    # overlay over original image
    #out = overlay * msk + img * (1.0 - msk)
    out = tf.math.add(tf.math.multiply(overlay, tmp_mask), tf.math.multiply(image, tf.math.subtract(tf.constant(1.0), tmp_mask)))
    out = tf.io.encode_jpeg(image=tf.image.convert_image_dtype(out, tf.uint8), format='rgb')
    #tf.print(image_path)
    tf.io.write_file('test.png', out)

    return image, mask, image_path


def process_path(image_path, mask_path):
    """ Reads images and masks based on their file paths. Has to be applied with tf.data.Dataset.map function
        Args:
        image_path: image path as string
        mask_path: mask path as string
        Returns:
        image, mask, image_path
    """
    img = tf.io.read_file(image_path)
    msk = tf.io.read_file(mask_path)
    img = decode_img(img, output_type='rgb')
    msk = decode_img(msk, output_type='rgb')
    return img, msk, image_path


def decode_img(img_path, output_type='rgb'):
    """ Decodes an tensor of type string to an float32 tensor. Has to be applied with tf.data.Dataset.map function
        Args:
        img: image as tensor of type string
        Returns:
        image as tensor of type float32
    """
    output_types = ['rgb', 'grayscale']
    assert output_type in output_types, 'output_type has to be in {}'.format(output_type)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img_path, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    if output_type == 'grayscale':
        img = tf.image.rgb_to_grayscale(img)
    return img


def mask_to_grayscale(image, mask, image_path):
    mask = tf.image.rgb_to_grayscale(mask)
    return image, mask, image_path


def random_flip(image, mask, image_path):
    """ Random flip images and masks. Has to be applied with tf.data.Dataset.map function
        Args:
        image: image as [heigth, width, channels]
        mask: mask as [heigth, width, channels]
        image_path: Path of image files. used to map images afterwards
        Returns:
        image, mask, image_path
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    return image, mask, image_path


def random_crop(image, mask, image_path):
    """ Random crops images and masks. Has to be applied with tf.data.Dataset.map function
        Args:
        image: image as [heigth, width, channels]
        mask: mask as [heigth, width, channels]
        image_path: Path of image files. used to map images afterwards
        Returns:
        image, mask, image_path
    """
    stacked_image = tf.stack([image, mask], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, Config.img_slice_height, Config.img_slice_width, 3])
    return cropped_image[0], cropped_image[1], image_path


def random_brightness(image, mask, image_path):
    """ Adds random brightness to images. Has to be applied with tf.data.Dataset.map function
        Args:
        image: image as [heigth, width, channels]
        mask: mask as [heigth, width, channels]
        image_path: Path of image files. used to map images afterwards
        Returns:
        image, mask, image_path
    """
    image = tf.image.random_brightness(image, 0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask, image_path


def central_crop(image, mask, image_path):
    """ Central Crops the images and masks with 64px border. Has to be applied with tf.data.Dataset.map function
        Args:
        image: image as [heigth, width, channels]
        mask: mask as [heigth, width, channels]
        image_path: Path of image files. used to map images afterwards
        Returns:
        image, mask, image_path
    """
    image = image[64:-64, 64:-64]
    mask = mask[64:-64, 64:-64]
    return image, mask, image_path


def add_gaussian_noise(image, mask, image_path):
    """ Adds gaussion noise to images. Has to be applied with tf.data.Dataset.map function
        Args:
        image: image as [heigth, width, channels]
        mask: mask as [heigth, width, channels]
        image_path: Path of image files. used to map images afterwards
        Returns:
        image, mask, image_path
    """
    if tf.random.uniform(()) > 0.5:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=(10) / (255), dtype=tf.float32)
        noise_img = image + noise
        noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
    else:
        noise_img = image
    return noise_img, mask, image_path
