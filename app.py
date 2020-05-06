from flask import Flask,jsonify,render_template
from flask import request
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import io
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app=Flask(__name__)
CORS(app)

def get_model():
    global model
    model=load_model("snake_mobilenet_10.h5")


get_model()

@app.route('/sample')
def running():
    return 'Flask is running'

# @app.route('/predict',methods=["POST"])
# def predict():   
#     message=request.get_json(force=True) 
#     name=message["name"]
#     address=message["address"]
#     person = {'name':name,'address':address}
#     return jsonify(person)


@app.route('/image',methods=["POST"])
def predict():
    try:
        message=request.get_json(force=True)  
        encoded=message["name"]
       
        decoded=base64.b64decode(encoded)
        image=Image.open(io.BytesIO(decoded))
        processedImage=preprocessImage(image,target_size=(96,96))
        X=processedImage/255 
       
        result=model.predict(X)
        number_to_class=["class-78","class-204","class-508","class-543","class-581","class-697","class-771","class-804","class-872","class-966"]
        index=np.argsort(result[0,:])
        result='Most likely class:'+str(number_to_class[index[9]])+'--probability:'+str(result[0,index[9]])
       
        person = {'result':result}
        return jsonify(person)
    except Exception as e:
        print(e)
        return "error"


def preprocessImage(image,target_size):
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    
    return image