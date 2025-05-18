import cv2
import face_recognition
import numpy as np
import os

# @uabali


dataset_path = "file path"
known_face_encodings = []
known_face_names = []

for person_dir in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_dir)
    if os.path.isdir(person_path):
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_dir)

# Web camera
cap = cv2.VideoCapture(0)

frame_count = 0
scale_factor = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()

    frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    frame_count += 1
    if frame_count % 3 != 0: 
        cv2.imshow('Yuz Tanima', original_frame)
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)

        cv2.rectangle(original_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(original_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', original_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
