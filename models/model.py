import typing

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU, GlobalAvgPool2D, Input

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

        self.top = self.top_model(top_input=self.backbone.output_shape)
        self.model = Sequential([
            self.backbone,
            self.top,
        ])

    @staticmethod
    def dense_block(x, units,  drop_rate=0.2):
        x = Dropout(drop_rate)(x)
        x = Dense(units)(x)
        x = ReLU()(x)
        return x

    def top_model(self, top_input):
        x_inp = Input(batch_shape=top_input)
        x = GlobalAvgPool2D()(x_inp)
        for units in self.num_hidden_units:
            x = self.dense_block(x, units)
        x = Dropout(0.2)(x)
        x_out = Dense(25, activation='softmax')(x)
        model = Model(inputs=x_inp, outputs=x_out, name=self.backbone_name + '_top')
        return model


if __name__ == '__main__':
    dm = DenseModel(
        num_hidden_units=(512,),
        backbone_name='densenet121',
        input_shape=(224, 224, 3),
        backbone_weights='imagenet')
