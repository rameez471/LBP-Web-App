from utils import *
from imutils.video import WebcamVideoStream
from flask import Response, Flask, render_template, request
import datetime
import imutils
import time
import threading
import cv2
from PIL import Image
from werkzeug.utils import secure_filename

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
uploads_path = os.path.join(APP_ROOT, 'datasets')
allowed_set = set(['png', 'jpg', 'jpeg'])
app.secret_key = os.urandom(24)

vs = WebcamVideoStream(src=0).start()
time.sleep(2.0)

faceObj = FaceObject()

@app.route('/')
def index_page():
    return render_template(template_name_or_list='index.html')


def detect_face_live():

    global outputFrame,vs, lock

    while True:

        frame = vs.read()
        frame = imutils.resize(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = faceObj.detect_face(gray)
        
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if faces is not None:
            for (x,y,w,h) in faces:

                image = gray[y:y+h, x:x+w]

                prediction = faceObj.recognize(image)

                if prediction is not None:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(frame,'%s' % (prediction),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                    cv2.putText(frame,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))

        outputFrame = frame.copy()


def detect_and_add():
    
    global outputFrame,vs, lock

    count = 0
    image_array = list()

    while count<30:

        frame = vs.read()
        frame = imutils.resize(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = faceObj.detect_face(gray)
        
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        if faces is not None:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                image = gray[y:y+h, x:x+w]
                image_array.append((image,count))

        count+=1

        outputFrame = frame.copy()

    return image_array



@app.route('/add_image',methods=['GET', 'POST'])
def add_image():
    images = detect_and_add()

    import matplotlib.pyplot as plt

    if request.method == 'POST':
        name = request.form['name']
        
        if name is  None:
            return render_template(
                template_name_or_list='warning.html',
                status='Enter Your Name'
            )

        for i in images:
            img = i[0]
            filename = name + str(i[1]) +'.jpg'
            filename = secure_filename(filename=filename)
            path = os.path.join(uploads_path,name)  

            save_image(img, filename, path)          


        return render_template(
            template_name_or_list='upload_result.html',
            status='Person is added successfully!!'
        )

    else:
        return render_template(
            template_name_or_list='warning.html',
            status='POST HTTP method required.'
        )


def generate():
    global outputFrame,lock

    while True:

        with lock:
            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode('.jpg',outputFrame)

            if not flag:
                continue

        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+
                bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_feed')
def face_detect_live():
    t = threading.Thread(target=detect_face_live)
    t.daemon = True
    t.start()
    return render_template('live.html')

@app.route("/predict")
def predict_page():
    """Renders the 'predict.html' page for manual image file uploads for prediction."""
    return render_template(template_name_or_list="predict.html")

@app.route('/video_feed_add')
def video_feed_add():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add')
def add_person():
    
    t = threading.Thread(target=detect_and_add)
    t.daemon = True
    t.start()
    return render_template(template_name_or_list="upload.html")


if __name__ == '__main__':

    app.debug = False
    app.run()

# release the video stream pointer
vs.stop()