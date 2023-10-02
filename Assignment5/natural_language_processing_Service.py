from flask import Flask
from flask import request
from urllib.parse import unquote
import os

from natural_language_processing import Sentiment

app = Flask(__name__)

# http://localhost:8786/infer?sentence="This place is really bad absolutly the worse"
# http://localhost:8786/infer?sentence="fantastic, great place I love it"

@app.route('/stats', methods=['GET'])
def getStats():
    return str(st.model_stats())

@app.route('/infer', methods=['GET'])
def getInfer():
    args = request.args
    sentence = args.get('sentence')
    sentence = unquote(sentence)
    return str(st.model_infer(sentence))

@app.route('/post', methods=['POST'])
def hellopost():
    args = request.args
    name = args.get('name')
    location = args.get('location')
    print("Name: ", name, " Location: ", location)
    imagefile = request.files.get('imagefile', '')
    print("Image: ", imagefile.filename)
    imagefile.save('/workspace/Hopkins/705.603Fall2023/workspace/ML_Microservice_Example/image.jpg')
    return 'File Received - Thank you'

if __name__ == "__main__":
    flaskPort = 8786
    st = Sentiment()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)

