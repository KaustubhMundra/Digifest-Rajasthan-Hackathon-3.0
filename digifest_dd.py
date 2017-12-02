import os 
import sys

# import dlib
# import imutils
import argparse

import time
# import cv2
#import playsound

from scipy.spatial import distance
from imutils import face_utils


from threading import Thread
from imutils.video import VideoStream

import numpy as np
import pandas as pd

from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame


pygame.init()


def eye_aspect_ratio(eye):

	one = distance.euclidean(eye[1], eye[5])
	two = distance.euclidean(eye[2], eye[4])
	three = distance.euclidean(eye[0], eye[3])
	
	ear = (one + two) / (2.0 * three)
	return ear
	
thresh = 0.25
frames = 20

detect = dlib.get_frontal_face_detector()

predict = dlib.shape_predictor("/Users/kaustubhmundra/Desktop/digifest/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap=cv2.VideoCapture(0)

flag=0

while True:

	ret, frame=cap.read()
	frame = imutils.resize(frame, width=1250)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	subjects = detect(gray, 0)

	for subject in subjects:

		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < thresh:

			flag += 1

			print (flag)

			if flag >= frames:

				cv2.putText(frame, "---------SLEEP----------IS-------DEATH---------WITHOUT-------INSOMNIATEC-------", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				


				pygame.mixer.music.load('/Users/kaustubhmundra/Desktop/digifest/Machine+Gun+4.wav')
				pygame.mixer.music.play(3)
				


				
				
		else:

			flag = 0

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (1000, 250),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

pygame.quit()
cv2.destroyAllWindows()

cap.stop()