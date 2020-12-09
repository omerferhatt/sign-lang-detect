import argparse

import tensorflow as tf
from dataset import Dataset
from models.mobilenetv2 import TSLMobilenetV2


def parser():
    pars = argparse.ArgumentParser(description='TSL Detection Model trainer')
    pars.add_argument('-d', '--data-path', type=str, required=True)
    pars.add_argument('-r', '--random-seed', type=int, default=10)
    pars.add_argument('-s', '--split-ratio', type=float, default=0.1)
    pars.add_argument('-b', '--batch-size', type=int, default=32)
    pars.add_argument('-e', '--epoch', type=int, default=25)
    pars.add_argument('-l', '--learning-rate', type=float, default=0.01)
    pars.add_argument('--debug', action='store_true')
    pars.add_argument('-c', '--checkpoint-dir', type=str, required=True)
    return pars.parse_args()


def main():
    dataset = Dataset(batch_size=arg.batch_size, split_ratio=arg.split_ratio, random_seed=arg.random_seed)
    train_files, eval_files = dataset.read_file_paths(arg.data_path)
    train_ds = dataset.create_dataset(train_files, is_training=True, debug=arg.debug)
    eval_ds = dataset.create_dataset(eval_files, is_training=True, debug=arg.debug)
    if arg.debug:
        dataset.check_ds(train_ds, image_count=5)
        dataset.check_ds(eval_ds, image_count=5)

    model = TSLMobilenetV2(input_shape=(224, 224, 3), num_classes=25, learning_rate=0.001)
    model.set_frozen_layers(level=319)
    model.summary()

    # y_pred = model.predict(tf.zeros((1, 224, 224, 3)))
    # model.fit(train_ds, epochs=arg.epoch, validation_data=eval_ds)
    # model.saved_model(arg.checkpoint_dir)


if __name__ == '__main__':
    arg = parser()
    main()
