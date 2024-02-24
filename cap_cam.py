# import the opencv library 
import os
import cv2
import time
import datetime
  
# Init
SET_CAM = True # Option to set camera
FRAME_WIDTH = 640 # (640,480), (1280,720), (1920, 1080)
FRAME_HEIGHT = 480 # 720
PROP_EXPOSURE = 400 # 400
PROP_AUTOFOCUS = 0 # 0,1
PROP_FOCUS = 100 # 0-255
dst_path = 'dst'

# define a video capture object 
cap = cv2.VideoCapture('/dev/v4l/by-id/usb-e-con_systems_See3CAM_130_2C234305-video-index0')
if not os.path.exists(dst_path):
    os.mkdir(dst_path)

# Set camera if SET_CAM=true
if SET_CAM:
    print('Setting Camera...')
    # Get params
    Width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    Height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    Framerate = cap.get(cv2.CAP_PROP_FPS)
    Brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    Contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    Saturation = cap.get(cv2.CAP_PROP_SATURATION)
    Gain = cap.get(cv2.CAP_PROP_GAIN)
    Hue = cap.get(cv2.CAP_PROP_HUE)
    Exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    AutoFocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    Focus = cap.get(cv2.CAP_PROP_FOCUS)

    if Width != FRAME_WIDTH:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        time.sleep(1)
        print(f'Set FRAME_WIDTH={FRAME_WIDTH} Done')

    if Height != FRAME_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        time.sleep(1)
        print(f'Set FRAME_HEIGHT={FRAME_HEIGHT} Done')
    
    if AutoFocus != PROP_AUTOFOCUS:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, PROP_AUTOFOCUS)
        print(f'Set PROP_AUTOFOCUS={PROP_AUTOFOCUS} Done')

    if Focus != PROP_FOCUS:
        cap.set(cv2.CAP_PROP_FOCUS, PROP_FOCUS)
        print(f'Set PROP_FOCUS={PROP_FOCUS} Done')

if __name__ == "__main__":
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
            print("AutoFocus = ",cap.get(cv2.CAP_PROP_AUTOFOCUS))
            print("Focus = ",cap.get(cv2.CAP_PROP_FOCUS))
            print("******************************")
        
        elif key == ord('c'):
            time_stamp = datetime.datetime.now().isoformat()
            cv2.imwrite(os.path.join(dst_path,time_stamp+'.jpg'),frame)
  
# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows()