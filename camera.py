import cv2
import torch
import numpy as np
from scipy import ndimage
from torchvision import transforms
from PIL import Image
from alexnet_model import Facial_Expression_Network_AlexNet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Facial_Expression_Network_AlexNet().to(device)
model.load_state_dict(torch.load("data/face_expression_model.pt"))
model.eval()

faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

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

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        #cv2.putText(img, "Douglas", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cropped = gray[y:y+h, x:x+w]
        crop_shape = cropped.shape

        # Make sure there is a face
        if crop_shape[0] < IMG_SIZE or crop_shape[1] < IMG_SIZE:
            continue

        b = cropped.tobytes()
        face_img = Image.frombuffer('L', (crop_shape[0], crop_shape[1]), b)
        face_img = transform1(face_img)
        face_img = face_img.squeeze(0)

        filtered = np.zeros_like(face_img)
        for cm in kirsch_compass_masks:
            k = ndimage.convolve(face_img, cm, mode='nearest', cval=0.0)
            filtered = np.add(filtered, k)

        filtered = transform2(filtered)
        filtered = filtered.unsqueeze(0).to(device)

        # Predict the expression
        out = model(filtered)
        with torch.no_grad():
            conf, pred = torch.max(out, 1)
        pred = pred.tolist()
        conf = conf.tolist()
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, _ = torch.max(probs, 1)
        conf = conf.tolist()
        cv2.putText(img, f"Conf: {int(conf[0]*100)}", (x+5, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, CLASSES[pred[0]], (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("video", img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()