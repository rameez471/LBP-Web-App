import numpy as np
import imutils
import cv2

class FaceObject:

    def __init__(self):
        self.haar_file = 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(self.haar_file)

    def detect_face(self,image):
        faces = self.face_detector.detectMultiScale(image,1.3,5)

        if len(faces) == 0:
            return None

        return faces