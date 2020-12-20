import os
import numpy as np
import tensorflow as tf
import cv2


# labels = sorted(os.listdir('data/Train'))
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', 'Z_DEL', 'Z_SPACE']
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
word_list = ''

cap = cv2.VideoCapture(0)
temp_list = []
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
    frame = cv2.putText(frame, word_list,
                        (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    if np.max(output_data.squeeze()) > 0.30:
        frame = cv2.putText(frame, str(labels[np.argmax(output_data).squeeze()]) + ' ' + str(np.max(output_data)),
                            (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        temp_list.append(labels[np.argmax(output_data).squeeze()])
        if len(list(set(temp_list[-15:]))) == 1 and len(temp_list) > 15:
            if temp_list[-1] == 'Z_SPACE':
                word_list += '_'
            elif temp_list[-1] == 'Z_DEL':
                word_list = word_list[:-1]
            else:
                word_list += temp_list[-1]
            temp_list = []
            # print(word_list)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('c'):
        word_list = ''
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
