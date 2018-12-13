from flask import Flask

app = Flask(__name__)
app.config.from_object('webapp.default_settings')

import webapp.views
# from model2 import *
