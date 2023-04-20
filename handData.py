import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import csv

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

record = False
Total_data = 1000
curr_data = 0
label = 4

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

# Initialize current frame
Curr_time = 0
Curr_frame_time = 0
prev_frame_time = 0

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()
    x, y, c = frame.shape

    #incresing the current time with 1
    Curr_time += 1

    # Initialize current time
    Curr_frame_time = time.time()

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

        	# Drawing landmarks on frames
        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

    # Display the FPS on the video screen
    fps = int(1/(Curr_frame_time - prev_frame_time))
    prev_frame_time = Curr_frame_time
    cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    # name of csv file 
    filename = "final_handData.csv"

    # In Capture interval, store the keypoints in csv file
    if cv2.waitKey(1)%256 == 32:
        record = True

    if record:
        #Adding the label
        landmarks.insert(0, label)

        # writing to csv file 
        with open(filename, 'a') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
            # writing the fields 
            csvwriter.writerow(landmarks)
            curr_data += 1
            csvfile.close()
        record = False

    if curr_data == Total_data:
        break
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
