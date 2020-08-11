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
    model=load_model("dense_model.h5")


get_model()

@app.route('/')
def running():
     return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    try:
        message=request.get_json(force=True)  
        encoded=message["image"]
       
        decoded=base64.b64decode(encoded)
        image=Image.open(io.BytesIO(decoded))
        processedImage=preprocessImage(image,target_size=(128,128))
        X=processedImage/255 
       
        result=model.predict(X)
        number_to_class=["class-78","class-204","class-508","class-543","class-581","class-697","class-771","class-804","class-872","class-966"]
        snake_names=["Timber rattlesnake","Common garter snake","Rough green snake",
        "Foxsnake","Eastern hognose snake","DeKay's brownsnake","Black rat snake","Copperhead","Western diamondback rattlesnake","Eastern racer"]

        index=np.argsort(result[0,:])
        results=[]     
        for x in range(9,0,-1):
            data={"imageClass":str(number_to_class[index[x]]),"snakeName":str(snake_names[index[x]]),"probability":str(result[0,index[x]])}
            results.append(data)
        
        print(results)
        return jsonify(results)
       
    except Exception as e:
        print(e)
        return "error"


def preprocessImage(image,target_size):
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    
    return image