
import cv2
import numpy as np
import os
import face_recognition
import math
import time


#Setting  list for the storing the data of face get detected
known_faces_list = []

# Haarcascade for the eye detection file HaarCascade_eye.xml
eye = cv2.CascadeClassifier('/home/vjae/PycharmProjects/persona_counter/venv/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')

#Initialize variables

face_locations = []
face_encodings = []

cap = cv2.VideoCapture(0)
frame_counter = 0
while True:
    ret,image = cap.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # general Image pre-processing
    image = cv2.flip(image, 1)
    rgb_frame = image[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    # if len(face_encodings)>0:
    #     face_encodings = face_encodings

    for top, right, bottom, left in face_locations:
        #calculating the area of the rectangle

        face_length = math.sqrt((bottom - left) ** 2)
        face_width = math.sqrt((right - top) ** 2)
        face_area = face_length * face_width

        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        roi_gray = gray[top:right,left:bottom]
        roi_color = image[top:right,left:bottom]
        eyes = eye.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_length = math.sqrt(((ex + ew) - ex) ** 2)
            eye_width = math.sqrt(((ey + eh) - ey) ** 2)
            eye_area = eye_length * eye_width

            #Eliminate false eye detected face to eye ratio is calculated
            face_eye_ratio = face_area / eye_area

            #Uncomment below line to for  ratio value
            # print ("face to eye ratio",ratio)

            #Limiting false face detection  taking ratio for certian values for better results

        #if (face_eye_ratio >= 8 and face_eye_ratio <= 50):
            cv2.rectangle(roi_color, (ex, ey), ((ex + ew), (ey + eh)), (0, 255, 0), 2)

            # Every Detected Face gets checked with list, if encoding is new then
            # increase person_count by one other wise pass.
            for face_encoding in face_encodings:
                #condition for 1st face
                 #face_encoding = face_encoding[:len(face_encoding)//16]
                 print(len(face_encoding),type(face_encoding))
                 if len(known_faces_list)==0:
                     known_faces_list.append(face_encoding)
                 else:
                    match_result = face_recognition.compare_faces(known_faces_list, face_encoding, tolerance=0.50)
                    print(match_result)
                    if True in match_result:

                        #print("match")
                        pass
                    else:
                        # print("unmatch)
                        known_faces_list.append(face_encoding)
# Printing count on the real time frame
    person_counter = str(len(known_faces_list))
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image,person_counter , (50,50), font, 0.5, (255, 0, 150), 2)
    cv2.imshow('image',image)
    frame_counter = frame_counter+1
# press escape or 'Q' to exit the code
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()