import os

import tensorflow as tf
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, batch_size=32, split_ratio=0.1, random_seed=1):
        """
        TODO: Add doc
        :param batch_size:
        :param split_ratio:
        :param random_seed:
        """
        self._batch_size = batch_size
        self._split_ratio = split_ratio
        self.__random_seed = random_seed

        self.labels = None

        if self.__random_seed > 0:
            tf.random.set_seed(self.__random_seed)

        self.autotune = tf.data.experimental.AUTOTUNE

    def read_file_paths(self, data_path: str):
        """
        TODO: Add doc
        :param data_path:
        :return:
        """
        self.labels = tf.io.gfile.listdir(data_path)
        path = os.path.join(data_path, '**/*.png')
        file_names = tf.io.gfile.glob(path)
        total_files = len(file_names)
        file_names = tf.random.shuffle(file_names)
        train_file_names = file_names[:int(total_files * (1 - self._split_ratio))]
        eval_file_names = file_names[int(total_files * (1 - self._split_ratio)):]
        return train_file_names, eval_file_names

    def create_dataset(self, files: list, is_training=False, debug=False):
        """
        TODO: Add doc
        :param files:
        :param is_training:
        :param debug:
        :return:
        """
        file_ds = tf.data.Dataset.from_tensor_slices(files)
        ds = file_ds.map(self.decode_files, num_parallel_calls=self.autotune)
        if debug:
            return ds
        if is_training:
            return ds.batch(self._batch_size).cache().prefetch(self.autotune)
        else:
            return ds.batch(self._batch_size)

    def check_ds(self, ds: tf.data.Dataset, image_count=3):
        """
        TODO: Add doc
        :param ds:
        :param image_count:
        :return:
        """
        np_iterator = self.get_numpy_iter(ds.take(image_count))
        fig, axes = plt.subplots(nrows=1, ncols=image_count)
        for ax, (image, label) in zip(axes, np_iterator):
            ax.imshow(image)
            ax.set_title(self.labels[label])
            ax.axis('off')
        plt.show()

    def decode_files(self, file):
        """
        TODO: Add doc
        :param file:
        :return:
        """
        image_byte = tf.io.read_file(file)
        image = tf.image.decode_png(image_byte)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.math.divide(tf.math.subtract(image, - tf.reduce_min(image)), tf.math.subtract(tf.reduce_max(image), - tf.reduce_min(image)))
        image = tf.subtract(tf.math.multiply(image, 2), 1)
        label = tf.strings.split(file, '/')[1]
        comparison = tf.cast(label == self.labels, dtype=tf.int64)
        label_id = tf.argmax(comparison)
        label_one_hot = tf.one_hot(label_id, len(self.labels))
        return image, label_one_hot

    @staticmethod
    def get_numpy_iter(ds: tf.data.Dataset):
        """Converts dataset object into numpy iterator"""
        return ds.as_numpy_iterator()

    @staticmethod
    def get_labels():
        tf.io.gfile.listdir()

    @property
    def batch_size(self):
        """Returns dataset's batch size"""
        return self._batch_size

    @property
    def split_ratio(self):
        """Returns dataset's split ratio"""
        return self._split_ratio

    @property
    def random_seed(self):
        """Returns dataset's random seed"""
        return self.__random_seed

    def set_batch_size(self, batch_size:int):
        self._batch_size = batch_size


if __name__ == '__main__':
    dataset = Dataset(batch_size=32, split_ratio=0.1, random_seed=3)
    dataset.set_batch_size(2)
    train_files, eval_files = dataset.read_file_paths('data')
    train_ds = dataset.create_dataset(train_files, debug=True)
    dataset.check_ds(train_ds, image_count=5)
