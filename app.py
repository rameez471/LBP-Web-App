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

# vs = VideoStream(src=0).start()
# time.sleep(2.0)

# @app.route('/')
# def index():
#     return render_template('index.html')



# def detect_face():
#     global vs, outputFrame, lock

#     face = FaceObject()
#     total = 0

#     while True:

#         frame = vs.read()
#         frame = imutils.resize(frame,width=800)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         faces = face.detect_face(gray)

#         timestamp = datetime.datetime.now()
#         cv2.putText(frame, timestamp.strftime(
# 			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
# 			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

#         if faces is not None:
#             for (x,y,w,h) in faces:
#                 cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

#                 image = gray[y:y+h, x:x+w]

#                 prediction = face.recognize(image)

#                 if prediction is not None:
#                     cv2.putText(frame,'%s - %.0f' % (prediction),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
#                 else:
#                     cv2.putText(frame,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

#         with lock:
#             outputFrame = frame.copy()


# def generate():

#     global outputFrame, lock

#     while True:
#         with lock:
#             if outputFrame is None:
#                 continue

#             (flag, encodedImage) = cv2.imencode('.jpg',outputFrame)

#             if not flag:
#                 continue

#         yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'+
#                 bytearray(encodedImage) + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     print('WTF')
#     return Response(generate(),
#             mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     	# construct the argument parser and parse command line arguments
# 	ap = argparse.ArgumentParser()
# 	ap.add_argument("-i", "--ip", type=str, required=True,
# 		help="ip address of the device")
# 	ap.add_argument("-o", "--port", type=int, required=True,
# 		help="ephemeral port number of the server (1024 to 65535)")
# 	args = vars(ap.parse_args())

# 	# start a thread that will perform motion detection
# 	t = threading.Thread(target=detect_face)
# 	t.daemon = True
# 	t.start()

# 	# start the flask app
# 	app.run(host=args["ip"], port=args["port"], debug=True,
# 		threaded=True, use_reloader=False)

# # release the video stream pointer
# vs.stop()

faceObj = FaceObject()

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

            faces = faceObj.detect_face(img)

            if faces is not None:
                for (x,y,w,h) in faces:
                    face = img[y:y+h, x:x+w]
                    save_image(img=face,filename=filename,uploads_path=uploads_path)

                    filename = remove_file_extension(filename=filename)


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
            
@app.route('/live',methods=['GET'])
def face_detect_live():
    try:
        cap = WebcamVideoStream(src=0).start()
        
        while True:
            fame_orig = cap.read()

            frame = cv2.resize(src=frame_orig, dsize=(0, 0), fx=0.5, fy=0.5)

            frame = frame[:, :, ::-1]

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            faces = faceObj.detect_face(frame)

            if faces is not None:
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                    face = frame[y:y+h, x:x+w]

                    prediction = faceObj.recognize(face)

                    if prediction is not None:
                        cv2.putText(frame,'%s - %.0f' % (prediction),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                    else:
                        cv2.putText(frame,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

                cv2.imshow(winname='Video', mat=frame_orig)
            else:
                continue

        cap.stop()
        cv2.destroyAllWindows()

        return render_template(template_name_or_list='index.html')

    except Exception as e:
        print(e)
        return render_template(
            template_name_or_list='warning.html',
            status = 'No face in database.'
        )

@app.route('/')
def index_page():
    return render_template(template_name_or_list='index.html')

@app.route('/predict')
def predict_page():
    return render_template(template_name_or_list='predict.html')

if __name__ == '__main__':

    app.run()