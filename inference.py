import os
import numpy as np
import tensorflow as tf
import cv2


# labels = sorted(os.listdir('data/Train'))
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', 'Z_DEL', 'Z_SPACE']
interpreter = tf.lite.Interpreter(model_path="saved_models/model_mobilenetv3_aug_86.tflite")
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

_, im = cap.read()
r = cv2.selectROI(im)
f = open('test.txt', 'a+')
f.write("\n")
f.close()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = rgb[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    rgb_resized = cv2.resize(rgb, (224, 224))
    normalized = rgb_resized[np.newaxis, :, :, :].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], normalized)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    frame_new = cv2.rectangle(frame, pt1=(r[0], r[1]), pt2=(r[0]+r[2], r[1]+r[3]), color=(0, 0, 255), thickness=1)
    frame_new = cv2.putText(frame_new, word_list,
                            (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    if np.max(output_data.squeeze()) > 0.40:
        frame = cv2.putText(frame, str(labels[np.argmax(output_data).squeeze()]) + ' ' + str(np.max(output_data)),
                            (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        temp_list.append(labels[np.argmax(output_data).squeeze()])
        if len(list(set(temp_list[-5:]))) == 1 and len(temp_list) > 5:
            f = open('test.txt', 'a+')
            if temp_list[-1] == 'Z_SPACE':
                word_list += ' '
                f.write(temp_list[-1])
            elif temp_list[-1] == 'Z_DEL':
                word_list = word_list[:-1]
                f.write(temp_list[-1])
            else:
                word_list += temp_list[-1]
                f.write(temp_list[-1])
            f.close()
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
