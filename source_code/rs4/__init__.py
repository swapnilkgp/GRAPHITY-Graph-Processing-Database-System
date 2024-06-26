from flask import Flask

app = Flask(__name__, template_folder='templates')

app.config['SECRET_KEY'] = 'ec9439cfc6c796ae2029594d'
from rs4 import routes