import os
from shutil import rmtree

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


class TSLMobilenetV2(tf.keras.models.Model):
    def __init__(self, input_shape, num_classes, learning_rate, *args, **kwargs):
        super(TSLMobilenetV2, self).__init__(*args, **kwargs)
        self.FREEZE_ALL = -1
        self.UNFREEZE_ALL = -2
        self._input_shape = input_shape
        self._num_classes = num_classes
        self._learning_rate = learning_rate
        self._custom_opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        self._custom_loss = tf.keras.losses.CategoricalCrossentropy()
        self._custom_metrics = [tf.keras.metrics.CategoricalAccuracy(),
                                tf.keras.metrics.TopKCategoricalAccuracy(3)]

        self.input_layer = layers.Input(shape=self._input_shape, name='tsl_input')
        self.augmentation_model = self.get_augmentation_model()
        self.sota_model = self.get_sota(input_shape=input_shape)

        self.avg_layer = layers.GlobalAveragePooling2D()
        self.b_norm = layers.BatchNormalization()
        self.out_drop = layers.Dropout(0.2)
        self.out_dense = layers.Dense(self._num_classes, activation='softmax')
        self.out = self.call(self.input_layer)
        super(TSLMobilenetV2, self).__init__(inputs=self.input_layer, outputs=self.out, *args, **kwargs)
        self.set_frozen_layers(level=200)
        self.compile(self._custom_opt, self._custom_loss, self._custom_metrics)

    @staticmethod
    def get_sota(input_shape=(224, 224, 3),
                 include_top=False,
                 weights='imagenet'):

        return tf.keras.applications.EfficientNetB1(
            input_shape=input_shape,
            include_top=include_top,
            weights=weights)

    def set_frozen_layers(self, level=200):
        total_layers = len(self.sota_model.layers)
        if level < total_layers or level > 0:
            for layer in self.sota_model.layers[:level]:
                layer.trainable = False
            for layer in self.sota_model.layers[level:]:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True
        elif level == -1:
            self.sota_model.trainable = False
        elif level == -2:
            self.sota_model.trainable = True
            for layer in self.sota_model.layers:
                if isinstance(layer, layers.BatchNormalization):
                    layer.trainable = False
        else:
            print(f'Percentage has to between 0 and 1. Selected percentage: {level}')
        self.compile(self._custom_opt, self._custom_loss, self._custom_metrics)

    @staticmethod
    def get_augmentation_model():
        return models.Sequential([
            layers.experimental.preprocessing.RandomWidth(0.2),
            layers.experimental.preprocessing.RandomHeight(0.2),
            layers.experimental.preprocessing.RandomRotation(0.3),
            layers.experimental.preprocessing.RandomZoom(0.2),
            layers.experimental.preprocessing.RandomContrast(0.1),
            layers.experimental.preprocessing.RandomFlip('vertical'),
            layers.experimental.preprocessing.Resizing(224, 224)
        ], name='tsl_augmentation')

    def get_config(self):
        pass

    def call(self, inputs, training=False, mask=None):
        x = inputs
        if training:
            x = self.augmentation_model(x, training=training)
        x = self.sota_model(x)
        x = self.avg_layer(x)
        x = self.b_norm(x)
        if training:
            x = self.out_drop(x, training=training)
        x = self.out_dense(x)
        return x

    def build(self, *args, **kwargs):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_encoder,
            outputs=self.out
        )

    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
        self._custom_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)


if __name__ == '__main__':
    model = TSLMobilenetV2(input_shape=(224, 224, 3), num_classes=25)
    model.summary()
