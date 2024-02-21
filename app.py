import time
import cv2
import torch
import numpy as np
import threading
import torchvision.transforms as transforms
from PIL import Image
from multiprocessing import Pool, Queue

# Init
q = Queue()
DIM = 224
cap = cv2.VideoCapture(0)

with open('models/classes.txt') as f:
    classes = f.readlines()
classes = list(classes)
for i in range(len(classes)):
    classes[i] = classes[i].replace('\n','')

# Normalize test set same as training set without augmentation
transform = transforms.Compose([transforms.Resize((DIM,DIM)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])

# Check cuda
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print('Testing on device: ', device)

# Load model
model = torch.load('models/checkpoint.pt', map_location='cpu')
model.eval()

# Define
def process():
    ret, frame = cap.read()

    while ret:
        ret, frame = cap.read()

        # Predict
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        im_pil = transform(im_pil)
        im_pil = im_pil.unsqueeze(0)
        output = model(im_pil)
        
        probs = torch.softmax(output, dim=1).detach().numpy()[0]
        for i, prob in enumerate(probs): probs[i] = round(prob, 3)
                
        pred_class = classes[torch.argmax(output, dim=1).detach().numpy()[0]]
        pred_prob = probs[torch.argmax(output, dim=1).detach().numpy()[0]]

        msg = f'Pred Class: {pred_class} Prob: {pred_prob}'
        print(msg)
          
        # Put to queue
        q.put(frame)

def Display():
    while True:
        if q.empty() != True:
            frame = q.get()
            
            cv2.imshow("frame1", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object 
    cap.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #  start threads
    p1 = threading.Thread(target=process)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()