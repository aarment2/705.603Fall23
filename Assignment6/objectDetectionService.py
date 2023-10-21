from flask import Flask
from flask import request
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
from wandb import Classes
from GraphicDataProcessing import ObjectDetection

app = Flask(__name__)

#Form to upload images
html_form_input = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Upload</title>
    </head>
    <body>
        <h1>Object Detection Assignment (Module 6)</h1>
        <form method="POST" enctype="multipart/form-data" action="/upload">
            <input type="file" name="image">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    """

#Form to display the output of the object detection
html_form_output = """
    <h3>Object Detection Completed</h3>
    <h3>Please see results in the objectDetectionResults directory</h3>
     """


#Define directory to upload images
UPLOAD_FOLDER = 'objectDetectionResults'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return html_form_input


@app.route('/upload', methods=['POST'])
def upload_image():
    # Check until file provided in the html form
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    # Validate that file not empty
    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to the defined UPLOAD_FOLDER
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        imagecv = cv.imread(filename)
        ot.process_image(imagecv,UPLOAD_FOLDER)
        
        return html_form_output


@app.route('/detect', methods=['POST'])
def detection():
    
   
    findings = ot.process_image()
    # covert to useful string
    findingsString = "Just testing for now"
    return findingsString

if __name__ == "__main__":
    flaskPort = 8786
    ot = ObjectDetection()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

