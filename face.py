import numpy as np
import imutils
import cv2
import os

class FaceObject:

    def __init__(self):
        self.haar_file = 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(self.haar_file)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.train()

    def detect_face(self,image):
        faces = self.face_detector.detectMultiScale(image)

        if len(faces) == 0:
            return None

        return faces

    def train(self):

        (images, labels, names, id) = ([], [], {}, 0)
        datasets = 'datasets'
        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    lable = id
                    images.append(cv2.imread(path, 0))
                    labels.append(int(lable))
                id += 1


        (images,labels) = [np.array(lis) for lis in [images,labels]]
        self.face_recognizer.train(images,labels)

    def recognize(self,image):
        
        prediction,confidence = self.face_recognizer.predict(image)
        print(confidence)
        if confidence < 50:
                return prediction
        
        return None

    
