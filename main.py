# To find point to point distances
import os
import time
from threading import Thread  # For multi-threading

import cv2  # For enabling computer vision
import dlib  # For face landmark detection
import imutils
import numpy as np
# For image processing on images
from imutils import face_utils
from imutils.video import VideoStream
from pygame import mixer
from scipy.spatial import distance as dist

mixer.init()


EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
eye_music = 0
ear_music = 0


def alarm():
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        mixer.music.load("drows.mp3")
        mixer.music.play()

    if alarm_status2:
        print('call')
        saying = True
        mixer.music.load("Yawn.mp3")
        mixer.music.play()
        saying = False

# Function to calculate eye aspect ratio


def eye_aspect_ratio(eye):

    upper_eyelid = dist.euclidean(eye[1], eye[5])
    lower_eyelid = dist.euclidean(eye[2], eye[4])

    medial = dist.euclidean(eye[0], eye[3])

    ear = (upper_eyelid + lower_eyelid) / (2.0 * medial)

    return ear

# Function to define shape of the eye


def final_ear(shape):
    # Left eye starting and ending point
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # right eye starting and ending point
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# Function to calculate upper and lower lip distance
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream().start()
time.sleep(1)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

    # Checking dowsiness state
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        # If closed eye counter exceeds threshold
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                eye_music = eye_music + 1
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm)
                    t.deamon = True
                    t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

    # checking Yawn state
        if (distance > YAWN_THRESH):
            ear_music = ear_music + 1
            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                t = Thread(target=alarm)
                t.deamon = True
                t.start()
        else:
            alarm_status2 = False

        # if eye_music >=25 or ear_music >= 20:
        #     mixer.music.play()
        #     eye_music = 0
        #     ear_music = 0

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
