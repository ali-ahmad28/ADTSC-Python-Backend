from ast import Global
from distutils.log import debug
from re import DEBUG, sub
import time
import datetime
from flask import Response , Flask, render_template
import os
import threading
from detection import ObjectDetection
from flask_cors import CORS
import cv2 
import json
from waitress import serve

app = Flask(__name__)
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
CORS(app)
# added to save video
label='starting detection'
flag = True
writer = ''
count=1
delete_count=1
initiateTraceBackIndex = 1
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
parent_directory = 'traceback'
traceback_counter  = 0



def generate_frames():
    print("Thread in main action", threading.get_ident())
    detection = ObjectDetection(0, "gunKnifeSmokeFire.pt")
    detection.start()
    now = datetime.datetime.now()
    finish_time = now + datetime.timedelta(seconds=20)
    delete_time = now + datetime.timedelta(seconds=40)
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # writer = cv2.VideoWriter('output.avi',fourcc, 30.0, (detection.width,detection.height))
    # print(detection.width,detection.height)
    # writer= cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*"x264"), 20, (detection.width,detection.height),True)
    while True:
            # if detection.stopped:
                # detection.release()
                # writer.release()
                # cv2.destroyAllWindows()
            #     break
                
            # else:
            #added for fps wait for time in miliseconds to execute following functionality 30 FPS only
            cv2.waitKey(detection.FPS_MS)
            real_frame = detection.read()
            
            
            
                #added to save video
            global flag,writer,count,delete_count,initiateTraceBackIndex
            if(flag==True):
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(f'output{count}.avi',fourcc, 30.0, (detection.width,detection.height))
                flag=False
                initiateTraceBackIndex=count-1
                count +=1
            if(datetime.datetime.now()<finish_time):
                writer.write(real_frame)
            else:
                finish_time = datetime.datetime.now() + datetime.timedelta(seconds=20)
                flag=True
            if(datetime.datetime.now()>=delete_time):
                try:
                    os.remove(f'output{delete_count}.avi')
                    delete_count +=1
                    delete_time = datetime.datetime.now() + datetime.timedelta(seconds=20) 
                except:
                    pass 
            
            
            results = detection.score_frame(detection.frame)
            global label,face_cascade,parent_directory,traceback_counter
            # labels,cord=results
            label=detection.getLabel(results)


            #added for trace back 

            print(label)
            if((label.__contains__('pistol') or label.__contains__('knife') or label.__contains__('smoke') or label.__contains__('fire'))and count>2):
                print(f'traceback started for {label}')
                directory = f'result{traceback_counter}'
                
                path  = os.path.join(parent_directory,directory)
                os.mkdir(path)
                # To use a video file as input 
                cap = cv2.VideoCapture(f'output{initiateTraceBackIndex}.avi')
                i=1
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
                        # cv2.imwrite('kang'+str(i)+'.jpg',img)
                        cv2.imwrite(f'traceback/result{traceback_counter}/{label}{i}.jpg',img)
                        #cv2.imwrite(f'traceback/{label}{i}.jpg',img)
                        # cv2.imshow('img', img)
                        i+=1
                    else:
                        print(f"trace back completed for {label}")
                        break
                traceback_counter +=1



            real_frame = detection.plot_boxes(results, detection.frame)
            real_frame = cv2.resize(real_frame, (640, 480))
            
            ret, buffer = cv2.imencode('.jpg', real_frame)
            frame = buffer.tobytes()

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/")
async def read_root():

    return "hello world"
@app.get("/api/anomalyType")
async def anomalyType():
        print(label)
        return label
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
serve(app,host='0.0.0.0',port=8080,threads=4)