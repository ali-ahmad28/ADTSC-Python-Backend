from re import DEBUG, sub
from flask import Response , Flask, render_template
import os
import threading
from detection import ObjectDetection
from flask_cors import CORS
import cv2
app = Flask(__name__)

CORS(app)

def generate_frames():
    print("Thread in main action", threading.get_ident())
    detection = ObjectDetection(0, "gunKnifeSmokeFire")
    detection.start()
    while True:
            if detection.stopped:
                break
                
            else:
                real_frame = detection.read()

            real_frame = cv2.resize(real_frame, (640, 640))
            ret, buffer = cv2.imencode('.jpg', real_frame)
            frame = buffer.tobytes()

            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/")
async def read_root():

    return render_template('index.html')


@app.get("/video")
async def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
