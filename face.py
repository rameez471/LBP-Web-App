import skimage
from os import path
from skimage.feature import local_binary_pattern
import cv2
import numpy as np

class Face:
    def __init__(self,app):
        self.storage = app.config['storage']
        self.db = app.db
        self.faces = []
        self.X = []
        self.Y = []
        self.face_user_keys = {}
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.load_all()
        self.train()

    def load_user_by_index_key(self, index_key=0):

        key_str = str(index_key)

        if key_str in self.face_user_keys:
            return self.face_user_keys[key_str]
        return None

    def load_train_file_by_name(self,name):
        trained_storage = path.join(self.storage,'trained')
        return path.join(trained_storage, name)

    def load_unknown_file_by_name(self, name):
        unknown_storage = path.join(self.storage,'unknown')
        return path.join(unknown_storage, name)

    def load_all(self):

        results = self.db.select("SELECT faces.id, faces.user_id, faces.filename, faces.created FROM faces")

        for row in results:

            user_id = row[1]
            filename = row[2]
            (width, height) = (130, 100)

            face = {
                "id": row[0],
                "user_id": user_id,
                "filename": filename,
                "created": row[3]
            }

            self.faces.append(face)
            face_image = cv2.imread(self.load_train_file_by_name(filename))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            # print(self.face_cascade.detectMultiScale(face_image))
            (x,y,w,h) = self.face_cascade.detectMultiScale(face_image)[0]
            face_image = face_image[y:y+h,x:x+w]
            # face_image = cv2.resize(face_image,(width,height))
            index_key = len(self.faces)
            self.X.append(face_image)
            self.Y.append(user_id)
            index_key_string = str(user_id)
            self.face_user_keys['{0}'.format(index_key_string)] = user_id 
        # X,Y = np.array(X),np.array(Y)
        # print('Training...')
        # print(Y)
        # if len(X):
        #     self.face_recognizer.train(X,Y)
        #     print('Model Trained!!')

    def train(self):
        print('Training...')
        if len(self.X):
            images, labels = np.array(self.X), np.array(self.Y)
            self.face_recognizer.train(images,labels)
            print('Model Trained!!')


    def recognize(self,unknown_filename):
        unknown_image = cv2.imread(self.load_unknown_file_by_name(unknown_filename))
        unknown_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2GRAY)
        (x,y,w,h) = self.face_cascade.detectMultiScale(unknown_image)[0]
        unknown_image = unknown_image[y:y+h,x:x+w]
        (width, height) = (130, 100)
        # unknown_image = cv2.resize(unknown_image,(width,height))
        
        prediction,confidence = self.face_recognizer.predict(unknown_image)

        if confidence < 80:
            user_id = self.load_user_by_index_key(prediction)
            return user_id
        return None





    