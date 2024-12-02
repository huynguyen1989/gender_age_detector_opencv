from flask import Flask, request, flash, redirect, render_template
import pdb

from utils import highlightFace
from nets import *

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = "./uploads/"

@app.route('/gender', methods=["GET"])
def upload_to_detect_gender():
   return render_template('gender_upload.html')

@app.route('/api/detect_gender', methods=["POST"])
def detect_gender():
    f = request.files['file']
    f.save(UPLOAD_FOLDER + f.filename)

    cap=cv2.VideoCapture(UPLOAD_FOLDER + f.filename)
    if not cap.isOpened():
        return("Error opening image file")
        
    hasFrame,frame=cap.read()
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    
    if not hasFrame:
        return("Error reading frame")

    if not faceBoxes:
        return("No face detected")
    
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')
    
    return {'Gender': gender}


@app.route('/age', methods=["GET"])
def upload_to_detect_age():
   return render_template('./age_upload.html')

@app.route('/api/detect_age', methods=["POST"])
def detect_age():
    f = request.files['file']
    f.save(UPLOAD_FOLDER + f.filename)

    cap=cv2.VideoCapture(UPLOAD_FOLDER + f.filename)
    if not cap.isOpened():
        return("Error opening image file")
        
    hasFrame,frame=cap.read()
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    
    if not hasFrame:
        return("Error reading frame")

    if not faceBoxes:
        return("No face detected")
    
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')
    
    return {'Age': age}
