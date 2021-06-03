import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image
from flask import Flask,render_template,Response

app=Flask(__name__)

FLAGS = []

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights',
	type=str,
	default='./yolov3.weights',
	help='Path to the file which contains the weights \
		 	for YOLOv3.')
parser.add_argument('-cfg', '--config',
	type=str,
	default='./cfg/yolov3.cfg',
	help='Path to the configuration file for the YOLOv3 model.')
parser.add_argument('-v', '--video-path',
	type=str,
	default='./Driving-Chinatown-SF.mp4',
	help='The path to the video file')
parser.add_argument('-vo', '--video-output-path',
	type=str,
    default='./output.mp4',
	help='The path of the output video file')
parser.add_argument('-l', '--labels',
	type=str,
	default='./coco-labels',
	help='Path to the file having the \
				labels in a new-line seperated way.')
parser.add_argument('-c', '--confidence',
	type=float,
	default=0.5,
	help='The model will reject boundaries which has a \
			probabiity less than the confidence value. \
			default: 0.5')
parser.add_argument('-th', '--threshold',
	type=float,
	default=0.3,
	help='The threshold to use when applying the \
			Non-Max Suppresion')
FLAGS, unparsed = parser.parse_known_args()
# Get the labels
labels = open(FLAGS.labels).read().strip().split('\n')
# Intializing colors to represent each label uniquely
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
# Load the weights and configutation to form the pretrained YOLOv3 model
net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)
# Get the output layer names of the model
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# If both image and video files are given then raise error
if FLAGS.video_path is None:
    print ('Path to video not provided')
# elif FLAGS.video_path:
	# Read the video
vid = cv.VideoCapture(0)#(str(FLAGS.video_path))
height, width, writer= None, None, None
def gen_frames():
	while True:
	    grabbed, frame = vid.read()
	    if not grabbed:
	    	break
	    if width is None or height is None:
	    	height, width = frame.shape[:2]
	    frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)
	    _,buffer=cv.imencode('.jpg',frame)
	    frame=buffer.tobytes()
	    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
			# if writer is None:
			# 	fourcc = cv.VideoWriter_fourcc(*'mp4v')
			# 	writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,(frame.shape[1], frame.shape[0]), True)


			# writer.write(frame)


		# print ("[INFO] Cleaning up...")
		# writer.release()
		# vid.release()


	# else:
	# 	print("[ERROR] Something's not right...")
