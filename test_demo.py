import torch
import face_recognition
import cv2
import numpy as np
from model import siameseResNet
from torchvision import transforms


test_images_path = [
    "images/obama.jpg",
    "images/biden.jpg",
    "images/pasha.jpg",
    "images/masha.jpg"
]

with torch.no_grad():
    model = siameseResNet()
    model.load_state_dict(torch.load(r"checkpoint\model2.pt", map_location=torch.device('cpu')))

    IMAGE_SHAPE = (128, 128)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.CenterCrop((90, 90)),
        transforms.Resize(IMAGE_SHAPE),
    ])
    # known_face_encodings = []
    faces = []
    for path in test_images_path:

        image = face_recognition.load_image_file(path)
        top, right, bottom, left = face_recognition.face_locations(image)[0]
        image = image[top:bottom, left: right]
        faces.append(transform(image))
    known_face_encodings = model.forward_img(torch.stack(faces)).numpy()

known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Pavlo Zakala",
    "Maria Zakala",
]

video_capture = cv2.VideoCapture(0)
with torch.no_grad():
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = []
        if len(face_locations) != 0:
            faces = torch.stack([transform(cv2.blur(rgb_frame[t:b, l: r].copy(), (5, 5))) for t, r, b, l in face_locations])
            face_encodings = model.forward_img(faces).numpy()

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            print(face_distances)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
