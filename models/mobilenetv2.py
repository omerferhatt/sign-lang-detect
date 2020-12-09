import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


class TSLMobilenetV2(tf.keras.models.Model):
    def __init__(self, input_shape, num_classes, *args, **kwargs):
        super(TSLMobilenetV2, self).__init__(*args, **kwargs)
        self._input_shape = input_shape
        self._num_classes = num_classes
        
        self._custom_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self._custom_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self._custom_metrics = [tf.keras.metrics.SparseCategoricalAccuracy(),
                                tf.keras.metrics.SparseTopKCategoricalAccuracy(3)]

        self.input_layer = layers.Input(shape=self._input_shape, name='tsl_input')
        self.augmentation_model = self.get_augmentation_model()
        self.mobilenet_v2_model = self.get_mobilenet(input_shape=input_shape)
        self.out_drop = layers.Dropout(0.5, name='out_dropout')
        self.out_dense = layers.Dense(self._num_classes, activation='softmax', name='out_dense')
        self.out = self.call(self.input_layer)
        super(TSLMobilenetV2, self).__init__(inputs=self.input_layer, outputs=self.out,
                                             name='TSL_Mobilenet_v2_Detect', *args, **kwargs)
        self.compile(self._custom_opt, self._custom_loss, self._custom_metrics)
        
    @staticmethod
    def get_mobilenet(input_shape=(224, 224, 3),
                      include_top=False, weights='imagenet',
                      pooling='avg'):
        return tf.keras.applications.MobileNetV2(
            input_shape,
            include_top=include_top,
            weights=weights,
            pooling=pooling)
    
    def set_frozen_layers(self, percentage=0.95):
        total_layers = len(self.mobilenet_v2.layers)
        if percentage < 1 or percentage > 0:
            for layer in self.mobilenet_v2.layers[:int(total_layers*percentage)]:
                layer.trainable = False
            for layer in self.mobilenet_v2.layers[int(total_layers*percentage):]:
                layer.trainable = True
        elif percentage == 1:
            self.mobilenet_v2.trainable = False
        elif percentage == 0:
            self.mobilenet_v2.trainable = True
        else:
            print(f'Percentage has to between 0 and 1. Selected percentage: {percentage}')
        self.compile(self._custom_opt, self._custom_loss, self._custom_metrics)

    @staticmethod
    def get_augmentation_model():
        return models.Sequential([
            layers.experimental.preprocessing.RandomWidth(0.1),
            layers.experimental.preprocessing.RandomHeight(0.1),
            layers.experimental.preprocessing.RandomRotation(0.3),
            layers.experimental.preprocessing.RandomZoom(0.1),
            layers.experimental.preprocessing.RandomContrast(0.1),
            layers.experimental.preprocessing.RandomFlip('vertical'),
            layers.experimental.preprocessing.Resizing(224, 224)
        ], name='tsl_augmentation')
    
    def get_config(self):
        pass
    
    def call(self, inputs, training=None, mask=None):
        x = self.augmentation_model(inputs)
        x = self.mobilenet_v2_model(x)
        x = self.out_drop(x)
        x = self.out_dense(x)
        return x

    def build(self, *args, **kwargs):
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_encoder,
            outputs=self.out
        )

    def saved_model(self, checkpoint_dir: str):
        tf.saved_model.save(self, checkpoint_dir)


if __name__ == '__main__':
    model = TSLMobilenetV2(input_shape=(224, 224, 3), num_classes=25)
    model.summary()
