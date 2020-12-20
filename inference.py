import os
import numpy as np
import tensorflow as tf
import cv2


labels = sorted(os.listdir('data/Train'))
labels_custom = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', 'Z_DEL', 'Z_SPACE']
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (224, 224))
    normalized = rgb_resized[np.newaxis, :, :, :].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], normalized)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.max(output_data.squeeze()) > 0.4:
        print(labels[np.argmax(output_data).squeeze()], np.max(output_data))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
