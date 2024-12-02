import cv2

faceProto="models/opencv_face_detector.pbtxt"
faceModel="models/opencv_face_detector_uint8.pb"
ageProto="models/age_deploy.prototxt"
ageModel="models/age_net.caffemodel"
genderProto="models/gender_deploy.prototxt"
genderModel="models/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

padding=20