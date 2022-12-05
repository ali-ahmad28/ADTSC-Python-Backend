from ast import Global
from flask import jsonify
from distutils.log import debug
from re import DEBUG, sub
import time
import datetime
from flask import Response , Flask, render_template
import os
import threading
from threading import Thread
from detection import ObjectDetection
from flask_cors import CORS
import cv2 
import json
from waitress import serve
import cloudinary
import cloudinary.uploader
from skimage.metrics import structural_similarity
import multiprocessing
import concurrent.futures
import numpy as np
import shutil


app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
CORS(app)
# added to save video
#label='starting detection'
tracebackCloud=''
detectionRecordCloud=''
label = ["anomaly",tracebackCloud,detectionRecordCloud]
flag = True
writer = ''
traceback_writer=''
count=1
delete_count=1
initiateTraceBackIndex = 1
finish_time=''
delete_time=''
similarityFlage = False
nextTraceBackTimer = ''

# finish_time = datetime.datetime.now() + datetime.timedelta(seconds=20)
# delete_time = datetime.datetime.now() + datetime.timedelta(seconds=40)
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
parent_directory = 'traceback'
traceback_counter  = 0

cloudinary.config(
    cloud_name = "dsycufi98",
    api_key = "748661129525595",
    api_secret = "50NZHFOz6gDGW0bgHRjQt9RWwnA"
)

def save_cloudinary(img):
    result= cloudinary.uploader.upload(img)
    url = result["secure_url"]
    return url
def save_cloudinary_video(video):
    result = cloudinary.uploader.upload(video,resource_type = "video")
    url = result["secure_url"]
    return url
def traceback(detection,detectedClass):
    global face_cascade,parent_directory,traceback_counter,count,initiateTraceBackIndex,tracebackCloud,detectionRecordCloud,similarityFlage,traceback_writer
    similarityFlage = False

    #added to clear old traceback record from server

    if(traceback_counter>0):
        shutil.rmtree(f'traceback/result{traceback_counter-1}')
    #if((label[0].__contains__('pistol') or label[0].__contains__('knife') or label[0].__contains__('smoke') or label[0].__contains__('fire'))and count>2):
    directory = f'result{traceback_counter}'
    print(f'traceback started for {detectedClass}')
    path  = os.path.join(parent_directory,directory)
    os.mkdir(path)
    # To use a video file as input 
    cap = cv2.VideoCapture(f'output{initiateTraceBackIndex}.avi')
    i=1
    #added for converting traceback images to video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    traceback_writer = cv2.VideoWriter(f'traceback/result{traceback_counter}/{detectedClass}.avi',fourcc, 5.0, (detection.width,detection.height))
    print(".........................")
    while True:
    # Read the frame
        grab, img = cap.read()
        if(grab):
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            imgC = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite('kang'+str(i)+'.jpg',img)
            #added for similarity check
            if(similarityFlage==True):
                img1 = cv2.imread(f'traceback/result{traceback_counter}/{detectedClass}{i-1}.jpg')
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                score, diff = structural_similarity(img1, imgC , full=True,)
                if(int(score*100)<=80):
                    cv2.imwrite(f'traceback/result{traceback_counter}/{detectedClass}{i}.jpg',img)
                    #added for converting traceback images to video
                    traceback_writer.write(img)
                    i+=1
            else:
                cv2.imwrite(f'traceback/result{traceback_counter}/{detectedClass}{i}.jpg',img)
                #added for converting traceback images to video
                traceback_writer.write(img)
                i+=1
                similarityFlage=True       
                # cv2.imwrite(f'traceback/result{traceback_counter}/{label[0]}{i}.jpg',img)
                #cv2.imwrite(f'traceback/{label}{i}.jpg',img)
                # cv2.imshow('img', img)
                # i+=1
        else:
            print(f"trace back completed for {detectedClass}")
            traceback_writer.release()
            break
    try:
        print("try")
        # tracebackCloud = save_cloudinary(f'traceback/result{traceback_counter}/{label[0]}{1}.jpg')
        # label[1]=tracebackCloud
        # print("saved to cloud",tracebackCloud)

        #added to upload video 
        #tracebackCloud = save_cloudinary_video(f'traceback/result{traceback_counter}/{detectedClass}.avi')
        label[1]=tracebackCloud
        print("saved traceback video to cloud",tracebackCloud)

        #added to save detection video to cloud

        #detectionRecordCloud = save_cloudinary_video(f'output{initiateTraceBackIndex}.avi')
        label[2]=detectionRecordCloud
        print("saved detection video to cloud",detectionRecordCloud)

    except:
        print("error cant save to cloud")
        pass
    traceback_counter +=1

def save_video(detection,real_frame):
    global flag,writer,count,delete_count,initiateTraceBackIndex,finish_time,delete_time
    if(flag==True):
        print("inside true")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(f'output{count}.avi',fourcc, 5.0, (detection.width,detection.height))
        flag=False
        initiateTraceBackIndex=count-1
        count +=1
    if(datetime.datetime.now()<finish_time):
        writer.write(real_frame)
    else:
        print("inside false")
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=20)
        flag=True
    if(datetime.datetime.now()>=delete_time):
        try:
            print("inside delete")
            delete_time = datetime.datetime.now() + datetime.timedelta(seconds=20)
            os.remove(f'output{delete_count}.avi')
            delete_count +=1
        except:
            pass

def generate_frames():
    print("Thread in main action", threading.get_ident())
    global finish_time,delete_time,nextTraceBackTimer
    finish_time = datetime.datetime.now() + datetime.timedelta(seconds=20)
    delete_time = datetime.datetime.now() + datetime.timedelta(seconds=40)
    nextTraceBackTimer = datetime.datetime.now()
    detection = ObjectDetection(0, "weights\Adtsc450.pt")
    detection.start()
    while True:
            #print("main")
            #added for fps wait for time in miliseconds to execute following functionality 30 FPS only
            cv2.waitKey(detection.FPS_MS)
            real_frame = detection.read()
            #added to save video
            save_video(detection,real_frame) 
            results = detection.score_frame(detection.frame)
            global label,face_cascade,parent_directory,traceback_counter,count
            #label[0]=detection.getLabel(results)
            anomaly=detection.getLabel(results)
            #added for trace back 
            #traceback(detection,label)
            #added for multi-threading
            if((anomaly.__contains__('gun') or anomaly.__contains__('knife') or anomaly.__contains__('smoke') or anomaly.__contains__('fire') or anomaly.__contains__('fight') or anomaly.__contains__('car-carsh'))and count>2 and (datetime.datetime.now()>=nextTraceBackTimer)):
                label[0]=anomaly
                #detectedClass = label[0]
                thread = Thread(target=traceback,args=(detection,anomaly,))
                thread.daemon=True
                thread.start()
                nextTraceBackTimer = datetime.datetime.now() + datetime.timedelta(seconds=120)
            #thread.join()
            #added for multiprocessing for traceback
            # p1 = multiprocessing.Process(target=traceback, args=(label, ))
            # p1.daemon=True
            # p1.start()
            # p1.join()
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     executor.map(traceback,label)

            real_frame = detection.plot_boxes(results, detection.frame)
            real_frame = cv2.resize(real_frame, (640, 480))
            
            ret, buffer = cv2.imencode('.jpg', real_frame)
            frame = buffer.tobytes()

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/api/anomalyType")
async def read_root():
    print(label[0])
    print(label[1])
    print(label[2])
    return jsonify({'label': label[0],'link':label[1],'detection':label[2]})
# @app.get("/api/test")
# async def test():
#           result = save_cloudinary_video('output2.avi')
#           print(result)
#           return 'hello'
# @app.route("/api/anomalyType")
# async def anomalyType():
#     def get_type():
#         while(True):
#             time.sleep(1)
#             yield f'data: {label} \n\n'
#     return Response(get_type(),mimetype='text/event-stream')

   


@app.route("/api/video")
async def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__=="__main__":
#     app.run(debug=True)
serve(app,host='0.0.0.0',port=8080,threads=10)