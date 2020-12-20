import typing

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU, GlobalAvgPool2D, Input, Conv2D, BatchNormalization

from models.backbone import Backbone


class DenseModel(Backbone):
    def __init__(self,
                 optimizer,
                 loss,
                 metrics,
                 num_hidden_units: typing.Union[None, tuple],
                 backbone_name: str,
                 input_shape: tuple,
                 backbone_weights: typing.Union[None, str],
                 backbone_trainable=False):
        super(DenseModel, self).__init__(backbone_name, input_shape, backbone_weights, backbone_trainable)
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.num_hidden_units = num_hidden_units

        self.top = self.top_model(top_input=self.backbone.output_shape)
        self.model = Sequential([
            self.backbone,
            self.top,
        ])

        self.model.compile(self.optimizer, self.loss, self.metrics)

    @staticmethod
    def dense_block(x, units,  drop_rate=0.2):
        x = Dropout(drop_rate)(x)
        x = Dense(units)(x)
        x = ReLU()(x)
        return x

    def top_model(self, top_input):
        x_inp = Input(batch_shape=top_input)
        x = GlobalAvgPool2D()(x_inp)
        if self.num_hidden_units is not None:
            for units in self.num_hidden_units:
                x = self.dense_block(x, units)
        x = Dropout(0.4)(x)
        x_out = Dense(25, activation='softmax')(x)
        model = Model(inputs=x_inp, outputs=x_out, name=self.backbone_name + '_top')
        return model

    def set_freeze(self, block_select, opt=None, loss=None, metrics=None):
        conv_layer_idx = [(idx, layer.name, layer.count_params()) for idx, layer in enumerate(self.model.layers[0].layers) if isinstance(layer, Conv2D)]
        if block_select == 0:
            for layer in self.model.layers[0].layers:
                layer.trainable = False
        elif block_select == -1:
            for layer in self.model.layers[0].layers:
                layer.trainable = True
        else:
            block_select = len(conv_layer_idx) - block_select + 1
            if block_select > len(conv_layer_idx):
                print('You passed max block count, max:', len(conv_layer_idx))
            else:
                for idx, layer in enumerate(self.model.layers[0].layers):
                    if idx < conv_layer_idx[block_select - 1][0]:
                        layer.trainable = False
                    else:
                        if isinstance(layer, BatchNormalization):
                            layer.trainable = False
                        else:
                            layer.trainable = True

        if opt is None:
            opt = self.optimizer
        if loss is None:
            loss = self.loss
        if metrics is None:
            metrics = self.metrics
        self.model.compile(opt, loss, metrics)


if __name__ == '__main__':
    opt = tf.keras.optimizers.Adam(learning_rate=0.02)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy(k=3)]

    dm = DenseModel(
        optimizer=opt,
        loss=loss,
        metrics=metrics,
        num_hidden_units=(512,),
        backbone_name='effnetb0',
        input_shape=(224, 224, 3),
        backbone_weights='imagenet',
        backbone_trainable=True)

    dm.set_freeze(0)
    dm.model.summary()
