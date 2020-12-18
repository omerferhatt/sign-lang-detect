import os
from functools import partial

import numpy as np
import tensorflow as tf
import cv2

from utils.augmentation import transforms


def decode_img(path: str, labels, scale=1. / 255, target_shape=(224, 224), is_training=False):
    img_bin = tf.io.read_file(path)
    label = tf.strings.split(path, '/')[-2]
    ext = tf.strings.split(path, '.')[-1]
    if ext == 'png':
        img = tf.image.decode_png(img_bin, channels=3)
    elif ext == 'jpg' or ext == 'jpeg':
        img = tf.image.decode_jpeg(img_bin, channels=3)
    else:
        img = tf.image.decode_png(img_bin, channels=3)
    img = tf.image.resize(img, target_shape)

    if is_training:
        img = aug_process(img)
    else:
        img = tf.cast(img, tf.float32)
        # img = tf.multiply(img, scale)
        # img = tf.divide(tf.subtract(img, tf.constant((0.485, 0.456, 0.406))), tf.constant((0.229, 0.224, 0.225)))

    if tf.reduce_max(img) <=1.0:
        img = tf.multiply(img, 255.)
    comparison = tf.cast(label == labels, dtype=tf.int64)
    label_id = tf.argmax(comparison)
    label_encoded = tf.one_hot(label_id, depth=len(labels))
    return img, label_encoded


def aug_fn(image):
    data = {"image": image}
    aug_data = transforms(**data)
    img = aug_data["image"]
    return img


def aug_process(image):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)
    return aug_img


class Dataset:
    def __init__(self, path: str, extension_type: str, num_parallel_calls: int, random_seed=1,
                 is_training=False, target_shape=(224, 224), scale=1. / 255):
        self.path = path
        self.extension_type = extension_type
        self.num_parallel_calls = num_parallel_calls
        self.random_seed = random_seed
        self.is_training = is_training
        self.target_shape = target_shape
        self.scale = scale

        if random_seed > 0:
            tf.random.set_seed(self.random_seed)
            self.shuffle = True
        else:
            self.shuffle = False

        self.labels = sorted(tf.io.gfile.listdir(self.path))
        self._ds = self.read_files()
        self.map_files(decode_img, in_place=True)

    def read_files(self):
        files = tf.io.gfile.glob(os.path.join(self.path, f'**/*.{self.extension_type}'))
        files = tf.random.shuffle(files)
        return tf.data.Dataset.from_tensor_slices(files)

    def map_files(self, func, in_place=False):
        if in_place:
            self._ds = self._ds.map(
                partial(func,
                        labels=self.labels,
                        scale=self.scale,
                        target_shape=self.target_shape,
                        is_training=self.is_training),
                self.num_parallel_calls)
        else:
            return self._ds.map(
                partial(func,
                        labels=self.labels,
                        scale=self.scale,
                        target_shape=self.target_shape,
                        is_training=self.is_training),
                self.num_parallel_calls)

    def get_ds(self):
        return self._ds


if __name__ == '__main__':
    dataset_train = Dataset('data/Train', 'png', num_parallel_calls=tf.data.experimental.AUTOTUNE,
                            is_training=True, target_shape=(224, 224))

    train_ds = dataset_train.get_ds()

    dataset_test = Dataset('data/Test', 'png', num_parallel_calls=tf.data.experimental.AUTOTUNE,
                           is_training=False, target_shape=(224, 224), scale=1. / 255)

    test_ds = dataset_test.get_ds()

    for im, l in train_ds.batch(1).take(20).as_numpy_iterator():
        im = im[0]
        im_temp = (cv2.cvtColor(im[0], cv2.COLOR_BGR2RGB))
        cv2.imshow('img', im_temp)
        key = cv2.waitKey(0)
        if key == ord('n'):
            continue
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()
