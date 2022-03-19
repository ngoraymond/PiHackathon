from typing import ItemsView
import torch
import sys
import os
import cv2
import pandas as pd

import smtplib, ssl
from email.message import EmailMessage

ROOT = os.path.dirname(os.path.abspath(__file__))

def sendmail(dest, item, count, toSend = None):
    msg = EmailMessage()
    msg['Subject'] = f'{count} instances of {item} is missing'
    msg['From'] = 'hackberrypi2022@gmail.com'
    msg['To'] = dest

    if toSend is not None:
        cv2.imwrite(ROOT + 'to_send.jpg', toSend)
        with open(ROOT + 'to_send.jpg', 'rb') as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype='image', subtype = 'jpg')

    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context = ssl.create_default_context()) as s:
        s.login(msg['From'], 'password') #remember to place correct password
        s.send_message(msg)

if __name__ == '__main__':
    input_email = input('Enter your email: ') #email to send to
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  #one second for small, 5 seconds for medium
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
                sendmail(input_email, item, 1, toSend = frame)
            elif items_tracked[item] > cur_seen[item]: #item has left the camera view
                print(f'{items_tracked[item] - cur_seen[item]} instances of {item} has left the camera view')
                sendmail(input_email, item, items_tracked[item] - cur_seen[item], toSend = frame)
            
        items_tracked = cur_seen

        

