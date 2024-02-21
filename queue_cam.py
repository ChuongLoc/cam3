import time
import cv2
import threading
from multiprocessing import Pool, Queue

# Init
q = Queue()

# Define
def process():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        q.put(frame)

def Display():
  while True:
    if q.empty() != True:
        frame = q.get()
        cv2.imshow("frame1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == '__main__':
    #  start threads
    p1 = threading.Thread(target=process)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()