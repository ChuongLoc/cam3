# import the opencv library 
import time
import cv2 
  
# define a video capture object 
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap = cv2.VideoCapture(0)
print(cap)
if cap == None:
    cap = cv2.VideoCapture(0)
time.sleep(2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
time.sleep(2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
time.sleep(2)
cap.set(cv2.CAP_PROP_EXPOSURE, 400) 
time.sleep(2)

# get define of video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'fps={fps}')

width_frame=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(f'width_frame={width_frame}')

height_frame=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f'width_frame={height_frame}')

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = cap.read()
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'): 
        break
    elif key == ord('g'):
        print("******************************")
        print("Width = ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Height = ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Framerate = ",cap.get(cv2.CAP_PROP_FPS))
        print("Brightness = ",cap.get(cv2.CAP_PROP_BRIGHTNESS))
        print("Contrast = ",cap.get(cv2.CAP_PROP_CONTRAST))
        print("Saturation = ",cap.get(cv2.CAP_PROP_SATURATION))
        print("Gain = ",cap.get(cv2.CAP_PROP_GAIN))
        print("Hue = ",cap.get(cv2.CAP_PROP_HUE))
        print("Exposure = ",cap.get(cv2.CAP_PROP_EXPOSURE))
        print("******************************")  
  
# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows()