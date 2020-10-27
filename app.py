from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import time
from db import Database
from face import Face

app = Flask(__name__)
app.config['file_allowed'] = ['image/png', 'image/jpeg','image/jpg']
app.config['storage'] = path.join(getcwd(),'storage')
app.db = Database()
app.face = Face(app)

def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)

def error_handle(error_message, status=500, mimetype='application/json'):
    return Response(json.dumps({'error':{'message':error_message}}), status=status, mimetype=mimetype)

def get_user_by_id(user_id):
    user = {}
    results = app.db.select(
        'SELECT users.id, users.name, users.created, faces.id, faces.filename, faces.created FROM users LEFT JOIN ON faces.user_id = users.id WHERE users.id = ?',
        [user_id]
    )

    index = 0
    for row in results:
        
        face = {
            'id':row[3],
            'user_id':row[4],
            'filename':row[5],
            'created':row[6],
        }

        if index == 0:
            user = {
                'id':row[0],
                'name':row[1],
                'created':row[2],
                'faces':[],
            }
        if row[3]:
            user['faces'].append(face)
        index += 1

    if 'id' in user:
        return user
    return None

def delete_user_by_id(user_id):
    app.db.delete('DELETE FROM users WHERE users.id = ?', [user_id])
    app.db.delete('DELETE FROM faces WHERE faces.id = ?', [user_id])

@app.route('/',methods=['GET'])
def page_home():
    return render_template('index.html')

@app.route('/api',methods=['GET'])
def homepage():
    output = json.dumps({'api':'1.0'})
    return success_handle(output)

@app.route('/api/train', methods=['POST'])
def train():

    output = json.dumps({'success':True})

    if 'file' not in request.files:
        return error_handle('Face image is required.')
    
    else:
        file = request.files['file']

        if file.mimetype not in app.config['file_allowed']:
            return error_handle('Only *.png , *.jpg and *.jpeg files allowed')
        else:
            name = request.form['name']
            filename = secure_filename(file.filename)
            trained_storage = path.join(app.config['storage'],'trained')
            file.save(path.join(trained_storage, filename))

            created = int(time.time())
            user_id = app.db.insert('INSERT INTO users(name, created) values(?,?)',[name,created])

            if user_id:
                face_id = app.db.insert('INSERT INTO faces(user_id, filename,created) values(?,?,?)'
                                        [user_id, filename,created])
                
                if face_id:
                    face_data = {'id': face_id,
                                'filename':filename,
                                'created':created}
                    return_output = json.dumps({'id':user_id,'name':name,'face':face_data})
                    return success_handle(return_output)
                else:
                    return error_handle('Error saving face image.')

            else:
                return error_handle('An Erro occured interting new user')

    return success_handle(output)

@app.route('/api/users/<int:user_id>',methods=['GET','DELETE'])
def user_profile(user_id):
    if request.method == 'GET':
        user = get_user_by_id(user_id)    
        if user:
            return success_handle(json.dumps(user),200)
        else:
            return error_handle('User not found',404)

    if request.method == 'DELETE':
        delete_user_by_id(user_id)
        return success_handle(json.dumps({'deleted':True}))


@app.route('/api/recognize',methods=['POST'])
def recognize():

    if 'file' not in request.files:
        return error_handle('Image is required.')
    else:
        file = request.files['file']
        
        if file.mimetype not in app.config['file_allowed']:
            return error_handle('File extension is not allowed.')
        else:
            filename = secure_filename(file.filename)
            unknown_storage = path.join(app.config['storage'],'unknown')
            file_path = path.join(unknown_storage, filename)
            file.save(filename)

            user_id = app.face.recognize(filename)
            if user_id:
                user = get_user_by_id(user_id)
                message = {'message':'Hey we found {0} math with your face image'.format(user['name']),
                            'user':user}
                return success_handle(json.dumps(message))
            else:
                return error_handle('Device Unlocked')


app.run()