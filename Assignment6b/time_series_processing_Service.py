from flask import Flask
from flask import request
import os

from time_series_processing import CombinedCardiacPressure, CarotidPressure, IlliacPressure

app = Flask(__name__)

# http://localhost:8786/infer?carotid_timeseries_filename=/FULL/PATH/TO/TIME/SERIES/FILE/carotid_pressure_test_1.csv&illiac_timeseries_filename="/FULL/PATH/TO/TIME/SERIES/FILE/illiac_pressure_test_1.csv


@app.route('/stats_combined', methods=['GET'])
def getStats_combined():
    return str(CP.model_stats())

@app.route('/stats_illiac', methods=['GET'])
def getStats_illiac():
    return str(ip.model_stats())

@app.route('/stats_carotid', methods=['GET'])
def getStats_carotid():
    return str(cp.model_stats())

@app.route('/infer_combined', methods=['GET'])
def getInfer_combined():
    args = request.args
    carotid = args.get('carotid_timeseries_filename')
    illiac = args.get('illiac_timeseries_filename')
    str1 = "Combined Model Inference:"
    str2 = str(CP.model_infer(carotid, illiac))    
    return str1 + str2 + age_decoder

@app.route('/infer_carotid', methods=['GET'])
def getInfer_carotid():
    args = request.args
    carotid = args.get('carotid_timeseries_filename')
    str1 = "Carotid Model Inference:"
    str2 = str(cp.model_infer(carotid))  
    return str1 + str2 + age_decoder

@app.route('/infer_illiac', methods=['GET'])
def getInfer_illiac():
    args = request.args
    illiac = args.get('illiac_timeseries_filename')
    str1 = "Illiac Model Decoding:"
    str2 = str(ip.model_infer(illiac))  
    return str1 + str2 + age_decoder


if __name__ == "__main__":
    flaskPort = 8786
    CP = CombinedCardiacPressure()
    ip = IlliacPressure()
    cp = CarotidPressure()
    age_decoder = "<br /> AGE GROUPS:<br /> 0= 20s<br /> 1= 30s<br /> 2=4 40s<br /> 3= 50s<br /> 4= 60s<br /> 5= 70s"
    print('starting server...')
    app.run(host = '0.0.0.0', debug=True, port = flaskPort)

