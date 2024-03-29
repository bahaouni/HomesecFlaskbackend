from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
import os
import smtplib
from email.mime.text import MIMEText
import logging
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)
logging.basicConfig(filename='app.log', level=logging.INFO)

images = []  # List to store uploaded images
classNames = []  # List to store corresponding names of the uploaded images
encode_list_known = []  # List to store face encodings of known faces

@app.route('/uploadPhotos', methods=['POST'])
def upload_photos():
    try:
        data = request.json
        if not data:
            return jsonify({"result": "error", "message": "No data received"})

        base64_images = data.get('base64_images')
        names = data.get('names')

        if not base64_images or not names:
            return jsonify({"result": "error", "message": "Base64 images or names missing in request"})

        global images, classNames, encode_list_known
        images.clear()  # Clear previous images
        classNames.clear()  # Clear previous names

        # Decode base64 images and convert them into PIL Image objects
        decoded_images = []
        for base64_image in base64_images:
            if ',' in base64_image:
                _, base64_image = base64_image.split(',')

            while len(base64_image) % 4 != 0:
                base64_image += '='

            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            decoded_images.append(image)

        # Update images and classNames
        for name, img in zip(names, decoded_images):
            images.append(np.array(img))
            classNames.append(name)

        # Update known face encodings
        encode_list_known = find_encodings(images)

        return jsonify({"result": "success", "message": "Photos and names received and decoded successfully"})

    except Exception as e:
        logging.error(f'Error processing request: {e}')
        return jsonify({"result": "error", "message": str(e)})

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

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        data = request.json
        base64_image = data.get('image')

        if base64_image is None:
            return jsonify({"result": "error", "message": "Image data not provided"})

        _, base64_image = base64_image.split(',')

        while len(base64_image) % 4 != 0:
            base64_image += '='

        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"result": "error", "message": "Failed to decode image"})

        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        
        recognized = "unknown"
        if face_encodings:  # Check if face encodings are found
            face_distances = face_recognition.face_distance(encode_list_known, face_encodings[0])
            if face_distances.any():  # Check if face_distances is not empty
                match_index = np.argmin(face_distances)
                matches = face_recognition.compare_faces(encode_list_known, face_encodings[0])

                if matches[match_index]:
                    name = classNames[match_index].upper()
                    logging.info(f'Face recognized: {name}')
                    if name == "ELON":
                        send_email("bahaouni1@gmail.com")
                    recognized = name

                    y1, x2, y2, x1 = face_locations[0]
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite("annotated_image.jpg", img)

        return jsonify({"result": "success", "recognized": recognized})

    except Exception as e:
        logging.error(f'Error processing request: {e}')
        return jsonify({"result": "error", "message": str(e)})

@app.route('/hello')
def hello():
    return jsonify({"result": "Hello, World!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
