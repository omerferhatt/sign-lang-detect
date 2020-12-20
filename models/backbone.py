import typing


class Backbone:
    def __init__(self, name: str, input_shape: tuple, weights: typing.Union[None, str], trainable: bool):
        """
        Imports required classification task model backbone from tensorflow applications
        For now there is only 3 options to choose:
            - 'densenet121'
            - 'mobilenetv3'
            - 'effnetb0'
        Pretrained weight selection is only limited with `None` and 'imagenet'.

        :param name: Select backbone names from upper list, it's not case sensitive.
        :param input_shape: Input shape of backbone model.
        :param weights: Specifies which pretrained model weights will be used. It can be None,
                        so model will be trainable from scratch.
        :param trainable: Shows that model weights will be trained or not.
        """
        assert name in ['densenet121', 'mobilenetv3', 'effnetb0']
        self.backbone_name = name
        assert len(input_shape) == 3
        self.input_shape = input_shape
        assert weights is None or weights == 'imagenet'
        self.backbone_weights = weights
        if self.backbone_weights is not None:
            self.backbone_trainable = trainable
        else:
            self.backbone_trainable = True

        self._backbone = self.import_backbone()
        if self._backbone is not None:
            self._backbone.trainable = self.backbone_trainable

    def import_backbone(self):
        """Imports library according to name attribute, it is not case sensitive."""
        if self.backbone_name.lower() == 'densenet121':
            from tensorflow.keras.applications import DenseNet121
            model = DenseNet121(
                include_top=False,
                weights=self.backbone_weights,
                input_shape=self.input_shape
            )

        elif self.backbone_name.lower() == 'mobilenetv3':
            from tensorflow.keras.applications import MobileNetV3Small
            model = MobileNetV3Small(
                include_top=False,
                weights=self.backbone_weights,
                input_shape=self.input_shape
            )

        elif self.backbone_name.lower() == 'effnetb0':
            from tensorflow.keras.applications import EfficientNetB0
            model = EfficientNetB0(
                include_top=False,
                weights=self.backbone_weights,
                input_shape=self.input_shape
            )
        else:
            model = None
        return model

    @property
    def backbone(self):
        return self._backbone


if __name__ == '__main__':
    # For test
    b = Backbone('effnetb0', input_shape=(224, 224, 3), weights='imagenet', trainable=False)
    m = b.backbone
