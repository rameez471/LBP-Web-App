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
        self.face_user_keys = {}
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.load_all()

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
        X = list()
        Y = list()

        for row in results:

            user_id = row[1]
            filename = row[2]

            face = {
                "id": row[0],
                "user_id": user_id,
                "filename": filename,
                "created": row[3]
            }

            self.faces.append(face)
            face_image = cv2.imread(self.load_train_file_by_name(filename))
            index_key = len(self.faces)
            X.append(face_image)
            Y.append(user_id)
            index_key_string = str(user_id)
            self.face_user_keys['{0}'.format(index_key_string)] = user_id 
        X,Y = np.array(X),np.array(Y)
        print('Training...')
        print(Y)
        if len(X):
            self.face_recognizer.train(X,Y)
            print('Model Trained!!')

    def add_person(self, image):
        

    def face_detect(self,image):
        faces = self.face_cascade.detectMultiScale(image)
        face_coor = [(x,y,w,h) for (x,y,w,h) in faces]
        return face_coor

    def recognize(self,image):
        prediction,confidence = self.face_recognizer.predict(image)

        if confidence < 80:
            user_id = self.load_user_by_index_key(prediction)
            print(confidence)
            return user_id
        return None





    