import os
import glob
from flask import Flask
from flask import jsonify
from flask import request, render_template

from webapp import app
#from model.util import *
from SigNet import main1, getpredictions

valid_mimetypes = ['image/jpeg', 'image/png', 'image/tiff']

global model
# def get_predictions(img_name):
#     #TODO
#     return {
#         "bboxes":
#         [
#             {"x1": 10, "x2": 50, "y1": 10, "y2": 50}
#         ],
#     }

@app.route('/home')
def home1():
    model = main1([])
    return render_template('welcome.html')

@app.route('/load')
def index():
    #model = main1([])
    return render_template('index.html')

    # return render_template('index.html')

from PIL import Image
import numpy as np
@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        custid = request.form['customer ID']
        #print ('action' ,request.form['usr'])
        
        if 'file' not in request.files:
            return jsonify({'error': 'no file'}), 400
        # Image info
        img_file = request.files.get('file')
        img_name = img_file.filename
        mimetype = img_file.content_type
        # Return an error if not a valid mimetype
        print (img_file)
        if mimetype not in valid_mimetypes:
            return jsonify({'error': 'bad-type'})
        # Write image to static directory
        #print (app.config['UPLOAD_FOLDER'])
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

        #img = open_image(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        # Run Prediction on the model
        results = getpredictions(img_name, custid)
        
        if(results == 1):
            results = " Original "
        if(results == 0):
            results = " Forgery "
        # Delete image when done with analysis
        #os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

        return jsonify(results)
    
@app.route('/upload', methods=['POST','GET'])
def upload():
    
    if request.method == 'POST':
        custid = request.form['customer ID']
        #print ('action' ,request.form['usr'])
        
        if 'file' not in request.files:
            return jsonify({'error': 'no file'}), 400
        # Image info
        img_file = request.files.get('file')
        img_name = img_file.filename
        mimetype = img_file.content_type
        # Return an error if not a valid mimetype
        print (img_file)
        if mimetype not in valid_mimetypes:
            return jsonify({'error': 'bad-type'})
        # Write image to static directory
        #print (app.config['UPLOAD_FOLDER'])
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

        #img = open_image(os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        # Run Prediction on the model
        results = insertTable(custid,img_name,os.path.join(app.config['UPLOAD_FOLDER'], img_name))
        
        if(results == 1):
            results = "Upload Successfully"
        if(results == 0):
            results = "Not "
        results = "Upload Successfully"
        # Delete image when done with analysis
        #os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

        return jsonify(results)
    else:
       return render_template('upload.html')
    
    
import sqlite3


def createconnection():
    con = sqlite3.connect('test8.db')
    cursor = con.cursor()
    return cursor

    
def insertTable(Signatureid, filename,picture_file):
    insert_query = """INSERT INTO dataset (ID, fileName,file) VALUES(?,?, ?)"""
    c = createconnection()
    with open(picture_file, 'rb') as picture_file:
        ablob = picture_file.read()
        c.execute(insert_query, (Signatureid, filename, ablob))
    c.connection.commit()
    
    
def get_file_from_db(customer_id):
	cursor = createconnection()
	select_fname = """SELECT file,fileName from dataset where ID = ?"""
	cursor.execute(select_fname, (customer_id,))
	item = cursor.fetchall()
	cursor.connection.commit()
	return item

CREATE_TABLE = """CREATE TABLE IF NOT EXISTS dataset  (ID TEXT,fileName TEXT, file BLOB)"""
cursor = createconnection()
cursor.execute(CREATE_TABLE)
cursor.connection.commit()
