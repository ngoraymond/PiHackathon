import torch
import sys
import os
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  
    webcam = cv2.VideoCapture(0)

    while True:
        #webcam data
        ret, frame = webcam.read()

        if not ret:
            print('Cannot capture from webcam')
            continue

        results = model(frame)

        results.print()