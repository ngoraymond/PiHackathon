from typing import ItemsView
import torch
import sys
import os
import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  
    webcam = cv2.VideoCapture(0)
    items_tracked = {}

    while True:
        #webcam data
        ret, frame = webcam.read()

        if not ret:
            print('Cannot capture from webcam')
            continue

        results = model(frame)

        results.print()

        pds = results.pandas().xyxy[0]

        #only get high confident estimate
        pds = pds[pds.confidence > 0.5]

        cur_seen = {}
        for item in pds.name:
            if item not in cur_seen:
                cur_seen[item] = 0
            cur_seen[item] = cur_seen[item] + 1

        for item in items_tracked.keys():
            if item not in cur_seen:
                print(f'{item} has left the camera view')
            elif items_tracked[item] > cur_seen[item]: #item has left the camera view
                print(f'{items_tracked[item] - cur_seen[item]} instances of {item} has left the camera view')
            
        items_tracked = cur_seen

        

