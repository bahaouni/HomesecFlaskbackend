from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
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
CORS(app)  # Enable CORS for your Flask app

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Load known faces and their encodings
path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)

for cl in personsList:
    curPerson = cv2.imread(os.path.join(path, cl))
    images.append(curPerson)
    classNames.append(os.path.splitext(cl)[0])
logging.info(f'Loaded {len(images)} known faces.')

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

encode_list_known = find_encodings(images)
logging.info('Encoding complete.')
@app.route('/uploadPhotos', methods=['POST'])
def upload_photos():
    try:
        # Get JSON data from request
        data = request.json

        if not data:
            return jsonify({"result": "error", "message": "No data received"})

        # Extract base64-encoded images and names from the data
        base64_images = data.get('base64_images')
        names = data.get('names')

        if not base64_images or not names:
            return jsonify({"result": "error", "message": "Base64 images or names missing in request"})

        # Decode base64 images and convert them into PIL Image objects
        decoded_images = []
        for base64_image in base64_images:
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            decoded_images.append(image)

        # Now you can perform any necessary processing with the decoded images and names
        # For example, you can save the images to disk and use them for face recognition
        # Ensure to integrate this data into your face recognition logic

        # Return success response
        return jsonify({"result": "success", "message": "Photos and names received and decoded successfully"})

    except Exception as e:
        logging.error(f'Error processing request: {e}')
        return jsonify({"result": "error", "message": str(e)})


@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        # Get base64 image from request JSON data
        data = request.json
        base64_image = data.get('image')

        if base64_image is None:
            return jsonify({"result": "error", "message": "Image data not provided"})

        # Extract only the Base64-encoded image data
        _, base64_image = base64_image.split(',')

        # Add padding to the Base64 string if needed
        while len(base64_image) % 4 != 0:
            base64_image += '='

        # Decode base64 image to bytes
        image_bytes = base64.b64decode(base64_image)

        # Convert bytes to NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode NumPy array to OpenCV image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if the image is empty
        if img is None:
            return jsonify({"result": "error", "message": "Failed to decode image"})

        # Resize the image
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        # Perform face recognition
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encode_list_known, face_encoding)
            face_distances = face_recognition.face_distance(encode_list_known, face_encoding)
            match_index = np.argmin(face_distances)

            if matches[match_index]:
                name = classNames[match_index].upper()
                logging.info(f'Face recognized: {name}')
                if name == "ELON":
                    # Send email to 'baha'
                    send_email("bahaouni1@gmail.com")
                    break

                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Save the annotated image (optional)
        cv2.imwrite("annotated_image.jpg", img)

        return jsonify({"result": "success", "recognized": name if matches[match_index] else "unknown"})

    except Exception as e:
        logging.error(f'Error processing request: {e}')
        return jsonify({"result": "error", "message": str(e)})
@app.route('/hello')
def hello():
    return jsonify({"result": "Hello, World!"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
