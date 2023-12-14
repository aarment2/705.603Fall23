from flask import Flask
from flask import request
from werkzeug.utils import secure_filename
import os
import numpy as np
from ivf_processing import IVF

app = Flask(__name__)

#Form to upload images
html_form_input = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Upload</title>
    </head>
    <body>
        <h1>System Project</h1>
        <h3>TOWARDS AI-ENABLED IN VITRO FERTILIZATION</h3>
        <h4>Antonio Armenta. December 12, 2023</h4>
        <br>
        <p>Please upload four datasets available at HealthData.gov (https://healthdata.gov/dataset/)</p>
        <p>2021 Final Assisted Reproductive Technology (ART) Patient and Cycle Characteristics</p>
        <p>2021 Final Assisted Reproductive Technology (ART) Success Rates</p>
        <p>2021 Final Assisted Reproductive Technology (ART) Services and Profiles</p>
        <p>2021 Final Assisted Reproductive Technology (ART) Summary</p>
        <br>
        <br>
        <form method="POST" enctype="multipart/form-data" action="/upload">         
            <label for="patient_cycles">Patient and Cycle Characteristics:&nbsp;&nbsp;</label>
            <input type="file" name="patient_cycles" id="patient_cycles">
            <br>
            <label for="services">Services and Profiles:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</label>
            <input type="file" name="services" id="patient_cycles">
            <br>
            <label for="success_rates">Success Rates:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</label>
            <input type="file" name="success_rates" id="success_rates">
            <br>
            <label for="summary">Summary:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</label>
            <input type="file" name="summary" id="summary">
            <br>
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
UPLOAD_FOLDER = 'datasets'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return html_form_input


@app.route('/upload', methods=['POST'])
def upload_datasets():
    # Check until file provided in the html form    
    required_files = ['patient_cycles', 'services', 'success_rates', 'summary']
    new_filenames = [('patient_cycles', '2021_patient_cycles.csv'),
        ('services', '2021_services.csv'),
        ('success_rates', '2021_success_rates.csv'),
        ('summary', '2021_summary.csv')
    ]
    for file_key in required_files:
        if file_key not in request.files:
            return f"No '{file_key}' file found in the request", 400

    # Validate that file not empty
    try:
        for file_key, new_filename in new_filenames:
            file = request.files[file_key]
            if file:
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
         
        ivf_instance.ingest()
        ivf_instance.model_learn()

        return ivf_instance.results
    
    except Exception as e:
        return f"An error occurred: {e}", 500


if __name__ == "__main__":
    flaskPort = 8786
    ivf_instance = IVF()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

