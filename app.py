from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import time


app = Flask(__name__)
app.config['file_allowed'] = ['image/png', 'image/jpeg','image/jpg']
app.config['storage'] = path.join(getcwd(),'storage')
