import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from utils.dataset import Dataset


input_shape = (224, 224, 3)
batch_size = 128

dataset_test = Dataset('data/Test', 'png', num_parallel_calls=tf.data.experimental.AUTOTUNE,
                       is_training=False, target_shape=input_shape[:2])
test_ds = dataset_test.get_ds()
test_ds = test_ds.cache().batch(798).prefetch(tf.data.experimental.AUTOTUNE)

m = tf.keras.models.load_model('saved_models/TSL_MobileNetV2_modified.h5')
y_pred = m.predict(test_ds)
y_pred = np.argmax(y_pred, axis=-1)
_, y = next(test_ds.take(1).as_numpy_iterator())
y = np.argmax(y, axis=-1)

print('Confusion Matrix')
print(confusion_matrix(y, y_pred))
print('Classification Report')
target_names = dataset_test.labels
print(classification_report(y, y_pred, target_names=target_names))