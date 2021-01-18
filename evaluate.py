import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
conf_res = confusion_matrix(y, y_pred)
print(conf_res)
print('Classification Report')
target_names = dataset_test.labels
print(classification_report(y, y_pred, target_names=target_names))

target_names[-2] = 'DEL'
target_names[-1] = 'SPACE'
fig, ax = plt.subplots(figsize=(10, 15))
im = ax.imshow(conf_res)
ax.set_xticks(np.arange(len(target_names)))
ax.set_yticks(np.arange(len(target_names)))
# ... and label them with the respective list entries
ax.set_xticklabels(target_names)
ax.set_yticklabels(target_names)
plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

for i in range(len(target_names)):
    for j in range(len(target_names)):
        text = ax.text(j, i, conf_res[i, j],
                       ha="center", va="center", color="w")

fig.tight_layout()
plt.show()
