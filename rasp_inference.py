import os
import time
from copy import deepcopy
from datetime import datetime
from typing import Union

import numpy as np
import tensorflow as tf
import cv2


class TSLReader:
    def __init__(self, labels_txt: str, model_path: str, camera_device=0):
        self.labels_txt = labels_txt
        self.model_path = model_path
        self.camera_device = camera_device

        self.labels = self.get_labels()
        self.interpreter, self.model_details = self.init_tensor()
        self.word_placeholder = ''
        self.temp_alpha = []

        self.capture = self.init_camera()
        self.frame_time = 0
        self.roi = None

    def init_tensor(self) -> Union[None, tuple]:
        try:
            t = time.time()
            print(f"{self.get_time_str()}-[I] Tensorflow Lite interpreted starting")
            interpreter = tf.lite.Interpreter(model_path=self.model_path, num_threads=8)
            print(f"{self.get_time_str()}-[I] Allocating tensors")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            model_details = {
                'input_details': input_details,
                'output_details': output_details,
                'input_shape': input_details[0]['shape']
            }
            print(f"{self.get_time_str()}-[I] Model initialize completed in {time.time() - t:0.3f} seconds")
            return interpreter, model_details
        except Exception as e:
            print(f"{self.get_time_str()}-[E] Model load failed")
            return None

    def get_labels(self) -> Union[None, list]:
        try:
            print(f"{self.get_time_str()}-[I] Reading Labels")
            f = open(self.labels_txt, mode='r', encoding='utf-8')
            labels_list = f.readlines()
            for idx, label in enumerate(labels_list):
                if '\n' in label:
                    labels_list[idx] = label[:-1]
            if len(labels_list) <= 0:
                raise Exception
            print(f"{self.get_time_str()}-[I] {len(labels_list)} label read in total")
            return labels_list
        except Exception as e:
            print(f"{self.get_time_str()}-[E] There is no label in {self.labels_txt}")
            return None

    def init_camera(self):
        capture = cv2.VideoCapture(self.camera_device)
        if capture.isOpened():
            print(f"{self.get_time_str()}-[I] Camera device opened at: {self.camera_device}")
            return capture
        else:
            print(f"{self.get_time_str()}-[W] Cannot open camera device at: {self.camera_device}")
            return None

    def read_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            print(f"{self.get_time_str()}-[W] Can't receive frame (stream end?). Exiting ...")
            exit()
        else:
            target_shape = tuple(self.model_details['input_shape'][1:-1])
            frame_org = deepcopy(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.roi is not None:
                frame_org = cv2.rectangle(frame_org,
                                          pt1=(self.roi[0], self.roi[1]),
                                          pt2=(self.roi[0] + self.roi[2], self.roi[1] + self.roi[3]),
                                          color=(0, 0, 255), thickness=1)
                frame = frame[int(self.roi[1]):int(self.roi[1] + self.roi[3]),
                              int(self.roi[0]):int(self.roi[0] + self.roi[2])]
            frame = cv2.resize(frame, target_shape)
            frame = frame[np.newaxis, :, :, :].astype(np.float32)
            return frame_org, frame

    def inference(self, array):
        self.interpreter.set_tensor(self.model_details['input_details'][0]['index'], array)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.model_details['output_details'][0]['index'])

    @staticmethod
    def get_time_str():
        now = datetime.now()
        return now.strftime("%H:%M:%S")

    def decode_prediction(self, pred, threshold=0.4):
        pred = pred.squeeze()
        max_pred = max(pred)
        if max_pred >= threshold:
            decoded = self.labels[np.argmax(pred).squeeze()]
            self.temp_alpha.append(decoded)
            self.fill_temp()
            return decoded
        else:
            return None

    def fill_temp(self):
        if len(list(set(self.temp_alpha[-5:]))) == 1 and len(self.temp_alpha) > 5:
            if self.temp_alpha[-1] == self.labels[-1]:
                self.word_placeholder += ' '
            elif self.temp_alpha[-1] == self.labels[-2]:
                self.word_placeholder = self.word_placeholder[:-1]
            else:
                self.word_placeholder += self.temp_alpha[-1]
            self.temp_alpha = []

    def set_bbox(self):
        org, _ = self.read_frame()
        self.roi = cv2.selectROI('Select Image BBox', org)
        cv2.destroyWindow('Select Image BBox')

    @staticmethod
    def show_fps(image, fps):
        return cv2.putText(image, fps, (30, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    def __call__(self, *args, **kwargs):
        t = time.time()
        org, frame = self.read_frame()
        pred = self.inference(frame)
        fps = f"FPS: {1/(time.time()-t):0.1f}"
        org = self.show_fps(org, fps)
        decoded_pred = self.decode_prediction(pred, threshold=kwargs['threshold'])
        print(self.word_placeholder)
        if kwargs['show']:
            cv2.imshow('Original image', org)
            cv2.imshow("Model input", frame[0].astype(np.uint8))
            key = cv2.waitKey(1)
            if key == ord('q'):
                print(f"{self.get_time_str()}-[I] Exiting ...")
                return False
            elif key == ord('c'):
                print(f"{self.get_time_str()}-[I] Cleaning")
                self.word_list = ''
        if kwargs['stream']:
            return org
        return True


if __name__ == '__main__':
    reader = TSLReader(
        labels_txt='labels.txt',
        model_path='saved_models/model_mobilenetv3_aug_86.tflite',
        camera_device=0
    )
    reader.set_bbox()

    while reader(show=True, threshold=0.4):
        pass
