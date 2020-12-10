import os

import numpy as np
import cv2
import tensorflow as tf


if __name__ == '__main__':
    labels = ['Z_SPACE', 'R', 'A', 'Z', 'P', 'C', 'T', 'J', 'G', 'Z_DEL', 'D', 'U', 'K', 'I', 'S', 'M', 'E', 'L', 'H', 'N', 'V', 'Y', 'F', 'O', 'B']
    model = tf.keras.models.load_model('checkpoints/model_93val/model_93val')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[np.newaxis, :, :, :]
            pred = model(img)
            pred = np.squeeze(pred.numpy())
            if pred[np.argmax(pred)] > 0.90:
                print(labels[np.argmax(pred)], pred[np.argmax(pred)])
