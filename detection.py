from multiprocessing import Lock
import cv2 
import time
from threading import Thread
import threading
import logging
import subprocess
import os
import mimetypes
import torch
import numpy as np


class ObjectDetection:

    PATH = "yolov5-master"

    def __init__(self, capture_index, model_name):
        
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names

        #for fps
        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.cap = self.get_video_capture()
        assert self.cap.isOpened()

        self.grabbed, self.frame = self.cap.read()
        #added for fps to read only 30 frames per second
        time.sleep(self.FPS)

        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)

        self.width= int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height= int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.stopped = True

        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        # self.stopped = False
        self.t.start()

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """

        # return cv2.VideoCapture(self.capture_index)
        capture = cv2.VideoCapture(self.capture_index)
        
        # default buffer size is 10 so latest frame will be displayed after 10 frames i.e. skip 10 frames in 1 second.
        # i set the buffer size to 2 so latest frame will be displayed after 2 frames i.e. skip 2 frames in 1 secons.
        # now gives latest frames after droping previous two frames because now we only reading 30 fps 
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return capture

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """

        if model_name:
            print("model name reached")
            global PATH
            model = torch.hub.load(self.PATH,
                                   'custom',
                                   path="gunKnifeSmokeFire.pt",
                                   source='local',
                                   force_reload=True
                                   )
            # model = torch.hub.load('ultralytics/yolov5','custom', path="gunKnifeSmokeFire.pt")
            model.eval()
            # print(model)
            print("DONE")
        else:
            model = torch.hub.load('ultralytics/yolov5',
                                   'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # print(labels)
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(
                    row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(
                    labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame
    
    def getLabel(self,results):
        labels, cord= results
        n = len(labels)
        type=''
        for i in range(n):
            type += ' '+self.class_to_label(labels[i])
        return type
    def update(self):
        print("%d : Thread in targeted thread action", threading.get_ident())
        while True:

            #  if self.stopped is false, then we are reading the next frame
            # if not self.stopped:
                self.grabbed, self.frame = self.cap.read()

                # if self.grabbed is False:
                #     print('[Exiting] No more frames to read')
                #     # self.stopped = True
                #     pass
            # else:
            #     break
                # self.cap.release()

    # method for returning latest read frame
    def read(self):
        return self.frame

    # method called to stop reading frames
    # def stop(self):
    #     self.stopped = True

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        return: void
        """