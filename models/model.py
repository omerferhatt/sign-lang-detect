import typing

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, ReLU, GlobalAvgPool2D

from models.backbone import Backbone


class DenseModel(Backbone):
    def __init__(self,
                 num_hidden_units: tuple,
                 backbone_name: str,
                 input_shape: tuple,
                 backbone_weights: typing.Union[None, str],
                 backbone_trainable=False):
        super(DenseModel, self).__init__(backbone_name, input_shape, backbone_weights, backbone_trainable)
        self.num_hidden_units = num_hidden_units
        self.x = self._backbone.output
        self.x = GlobalAvgPool2D()(self.x)

        for units in self.num_hidden_units:
            self.x = self.dense_block(self.x, units)

        self.model = Model(inputs=self._backbone.inputs, outputs=self.x, name=self.backbone_name+'_dense_model')

    @staticmethod
    def dense_block(x, units,  drop_rate=0.2):
        x = Dropout(drop_rate)(x)
        x = Dense(units)(x)
        x = ReLU()(x)
        return x


if __name__ == '__main__':
    dm = DenseModel(
        num_hidden_units=(512, 256, 128, 64, 25),
        backbone_name='mobilenetv3',
        input_shape=(224, 224, 3),
        backbone_weights='imagenet')
