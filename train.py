import tensorflow as tf

from utils.dataset import Dataset
from models.model import DenseModel


input_shape = (224, 224, 3)
target_shape = input_shape[:2]
rescale = 1. / 255


epoch = 50
batch_size = 32
learning_rate = 1e-3


dataset_train = Dataset('data/Train', 'png', num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        is_training=True, target_shape=target_shape)
dataset_test = Dataset('data/Test', 'png', num_parallel_calls=tf.data.experimental.AUTOTUNE,
                       is_training=False, target_shape=target_shape, scale=rescale)

train_ds = dataset_train.get_ds()
test_ds = dataset_test.get_ds()

train_ds = train_ds.shuffle(512).batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
test_ds = test_ds.shuffle(512).batch(batch_size)

dm = DenseModel(
    num_hidden_units=(512, 25),
    backbone_name='mobilenetv3',
    input_shape=input_shape,
    backbone_weights='imagenet',
    backbone_trainable=False
)

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

dm.model.compile(optimizer=opt,
                 loss=tf.keras.losses.CategoricalCrossentropy(),
                 metrics=[tf.keras.metrics.CategoricalAccuracy()])

dm.model.fit(train_ds, validation_data=test_ds, epochs=epoch)
