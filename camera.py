"""
Code based on the following tutorial:
https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348

"""

import cv2
import torch
import numpy as np
from scipy import ndimage
from torchvision import transforms
from PIL import Image
from alexnet_model import Facial_Expression_Network_AlexNet

# Load the FER model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Facial_Expression_Network_AlexNet().to(device)
model.load_state_dict(torch.load("data/face_expression_model.pt"))
model.eval()

# Load the face recognition model
faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Get the webcam
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

# Global variables
IMG_SIZE = 96
BATCH_SIZE = 32
CLASSES = ['neutral', 'happy', 'surprised', 'sad', 'angry'] #, 'disgusted', 'afraid'
NUM_CLASSES = len(CLASSES)

transform1 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

transform2 = transforms.Compose([
    transforms.ToTensor()
])

kirsch_compass_masks = [
    np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]]),   # north
    np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]]),   # north-east
    np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]]),   # east
    np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]]),   # south-east
]

# Start video capture
while True:
    # Read from the webcamera
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscale

    # Detect the faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw a rectangle
        cropped = gray[y:y+h, x:x+w] # Get the area of the face
        crop_shape = cropped.shape

        # Make sure there is a face that is large enough
        if crop_shape[0] < IMG_SIZE or crop_shape[1] < IMG_SIZE:
            continue

        # Make the image the right size
        b = cropped.tobytes()
        face_img = Image.frombuffer('L', (crop_shape[0], crop_shape[1]), b)
        face_img = transform1(face_img)
        face_img = face_img.squeeze(0)

        # Filter the image
        filtered = np.zeros_like(face_img)
        for cm in kirsch_compass_masks:
            k = ndimage.convolve(face_img, cm, mode='nearest', cval=0.0)
            filtered = np.add(filtered, k)
        filtered = transform2(filtered)
        filtered = filtered.unsqueeze(0).to(device)

        # Predict the expression
        out = model(filtered)
        with torch.no_grad():
            _, pred = torch.max(out, 1)
        
        # Get the predictions and the confidence
        pred = pred.tolist()[0]
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, _ = torch.max(probs, 1)
        conf = conf.tolist()[0]

        # Print the results to the camera feed
        cv2.putText(img, f"Conf: {int(conf*100)}%", (x+5, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, CLASSES[pred], (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw the camera image
    cv2.imshow("video", img)

    # press "ESC" to quit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()