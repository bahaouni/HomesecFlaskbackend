import cv2
import numpy as np
import face_recognition
import os
import smtplib
from email.mime.text import MIMEText

path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)

for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
    print(os.path.splitext(cl)[0])
print(classNames)

def send_email(to_email):
    subject = "Face Recognition Alert"
    body = f"Face recognized: {to_email}"
    sender_email = "bahaouni47@gmail.com"  
    sender_password = "baHA0123*"  


    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_email

    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, "relo rrjn txck ikyl")
        server.sendmail(sender_email, to_email, msg.as_string())

def findEncodeings(image):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodeings(images)
print('Encoding Complete.')

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

    for encode_face, face_loc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encode_face)
        face_dis = face_recognition.face_distance(encodeListKnown, encode_face)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = classNames[match_index].upper()
            

            if name == "ELON":
                # Send email to 'baha'
                send_email("bahaouni1@gmail.com")
                print(name)
                break

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow('Face Recognition', img)
    cv2.waitKey(1)
