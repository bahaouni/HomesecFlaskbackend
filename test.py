import face_recognition
image = face_recognition.load_image_file("baha.png")
face_locations = face_recognition.face_locations(image)
face_landmarks_list = face_recognition.face_landmarks(image)
print(face_locations )
print(face_landmarks_list.left_eyebrow)