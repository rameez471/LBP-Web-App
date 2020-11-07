from utils import *
from imutils.video import WebcamVideoStream
from flask import Response, Flask, render_template, request
import threading
import argparse
import datetime
import imutils
import time
import cv2
from PIL import Image
from waitress import serve
from werkzeug.utils import secure_filename

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
uploads_path = os.path.join(APP_ROOT, 'uploads')
allowed_set = set(['png', 'jpg', 'jpeg'])


faceObj = FaceObject()

@app.route('/')
def index_page():
    return render_template(template_name_or_list='index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_feed')
def live_feed():
    return render_template('live.html')


@app.route('/predict')
def predict_page():
    return render_template(template_name_or_list='predict.html')

@app.route('/upload',methods=['POST','GET'])
def get_image():

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list='warning.html',
                status='No file field in POST request'
            )

        file = request.files['file']
        filename = file.filename

        if filename == '':
            return render_template(
                template_name_or_list='warning.html',
                status='No file selected!'
            )
        
        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            filename = secure_filename(filename=filename)
            img = np.asarray(Image.open(file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceObj.detect_face(img)

            if faces is not None:
                for (x,y,w,h) in faces:
                    face = img[y:y+h, x:x+w]
                    save_image(img=face,filename=filename,uploads_path=uploads_path)

                    filename = remove_file_extension(filename=filename)

                faceObj.train()
                return render_template(
                    template_name_or_list='upload_result.html',
                    status='Image uploaded successfully'
                )

            else:
                return render_template(
                    template_name_or_list='upload_result.html',
                    status='Image upload was unseccesfull! No face detected.'
                )

    else:
        return render_template(
            template_name_or_list='warning.html',
            status='POST http method required.'
        )

@app.route('/predictImage',methods=['GET', 'POST'])
def predict_image():

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list='warning.html',
                status='No file in POST request'
            )

        file = request.files['file']
        filename = file.filename

        if filename == '':
            return render_template(
                template_name_or_list='warning.html',
                status='No selected file!'
            )

        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            img = img = np.asarray(Image.open(file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceObj.detect_face(img)

            if faces is not None:
                (x,y,w,h) = faces[0]
                face = img[y:y+h, x:x+w]
                prediction = faceObj.recognize(face)
                if prediction is not None:
                    return render_template(
                        template_name_or_list='predict_result.html',
                        identity=prediction
                    )
                else:
                    return render_template(
                        template_name_or_list='predict_result.html',
                        identity='No identity found!'
                    )
            else:
                return render_template(
                    template_name_or_list='predict_result.html',
                    identity='Operation unseccesfull! No human face detected!'
                )

    else:
        return render_template(
            template_name_or_list='warning.html',
            status='POST HTTP method required.'
        )


def detect_face_live():
    global vs, outputFrame, lock


    vs = WebcamVideoStream(src=0).start()
    time.sleep(2.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame,width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceObj.detect_face(gray)

        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        if faces is not None:
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                image = gray[y:y+h, x:x+w]

                prediction = faceObj.recognize(image)

                if prediction is not None:
                    cv2.putText(frame,'%s' % (prediction),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                else:
                    cv2.putText(frame,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

        with lock:
            outputFrame = frame.copy()


def generate():

    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode('.jpg',outputFrame)

            if not flag:
                continue

        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+
                bytearray(encodedImage) + b'\r\n')


if __name__ == '__main__':

    t = threading.Thread(target=detect_face_live)
    t.daemon = True
    t.start()

    app.run()

# release the video stream pointer
vs.stop()