import numpy as np
import cv2
from scipy.misc import imresize
import time
import sys

position = (10,50)
# import serial

#ser = serial.Serial("COM22",115200)

from tensorflow.keras.models import load_model


# Load Keras model
model = load_model('Masrur.h5')

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.

     """

    # Get image ready for feeding into model

video_capture = cv2.VideoCapture("rain.mp4")

video_capture.set(3, 640)
video_capture.set(4, 480)
fps = video_capture.get(cv2.CAP_PROP_FPS)
print("a:",fps)

     
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('rain.avi' , fourcc , 20.0 , (640,480))
count = 0




    
while(video_capture.isOpened()):
   lanes = Lanes()


 

    # Capture the frames
   ret,image = video_capture.read()
   
   if ret:
           small_img = imresize(image, (80, 160, 3))
           small_img = np.array(small_img)
           #print(small_img.shape)
           small_img = small_img[None,:,:,:]
           # Make prediction with neural network (un-normalize value by multiplying by 255)
           prediction = model.predict(small_img)[0] * 255
           a = np.array(prediction[0:80,0:160,0])
           for i in range(0, 159):
                 if a[79,i]>60:
                    if a[79,i+1]>60 and a[79,i+2]>60:
                        point1=i
                        break
           for j in range(159, 2, -1):
                 if a[79,j]>60:
                    if a[79,j-1]>60 and a[79,j-2]>60:
                        point2=j
            
                        break
        
           point1=point1*4
           point2=point2*4
           print("point1:",point1)
           print("point2:",point2)
   #cv2.imwrite("frame%d.jpg" % count, prediction) 
   #count += 
   

    # Add lane prediction to list for averaging
           lanes.recent_fit.append(prediction)
     # Only using last five for average
           if len(lanes.recent_fit) > 5:
              lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
           lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
           blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
           lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
           lane_image = imresize(lane_drawn, (480, 640, 3))
           crop_img = lane_image[300:480, point1:point2]
           gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
           #cv2.imshow("c",gray)
           blur = cv2.GaussianBlur(gray,(5,5),0.1)
           #cv2.imshow("c",blur)
           ret,thresh = cv2.threshold(blur,60,255,cv2.THRESH_BINARY)
           _,contours,hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)
           # Find the biggest contour (if detected)
           if len(contours) > 0:
               c = max(contours, key=cv2.contourArea)

               M = cv2.moments(c)
               cx = int(M['m10']/M['m00'])
               cy = int(M['m01']/M['m00'])
               print("cx:",cx)
               print("cy:",cy)
               cv2.line(crop_img,(cx,0),(cx,480),(255,0,0),3)
               cv2.line(crop_img,(point1,cy),(point2,cy),(255,0,0),3)
               cv2.drawContours(crop_img, contours, -1, (0,255,0), 1)
               if cx >point1 and cx<point2:

                    text="On Track!"
                   # ser.write(str.encode('a'))
               if cx >= point2 :

                    text="Turn Left!"
               #     # ser.write(str.encode('b'))
               if cx <=point1:

                     text="Turn Right"
                    # ser.write(str.encode('c'))
               else:
                   print("Stop")
       # ser.write(str.encode('d'))

               
   
  
               

   
  
  
    # Merge the lane drawing onto the original image
          
   
  

   

 
               
   
   
   

           result = cv2.addWeighted(image, 1, lane_image, 1, 0)
           Masrur=cv2.putText(
      result, #numpy array on which text is written
      text, #text
      position, #position at which writing has to start
      cv2.FONT_HERSHEY_SIMPLEX, #font family
      1, #font size
      (209, 80, 0, 255), #font color
      3) #font stroke
           out.write(result)
           cv2.imshow('r',result)
  
  
           if cv2.waitKey(1) & 0xFF == ord('q'):

                 break
       
          #cv2.imshow("l",lane_image)
            

          #      
          #      # cv2.line(crop_img,(420,point1),(420,point2),(255,0,0),1)
          #      # cv2.line(crop_img,(479,point1),(479,point2),(255,0,0),1)
          #      # cv2.line(crop_img,(420,point1),(479,point1),(255,0,0),1)
          #      # cv2.line(crop_img,(420,point2),(479,point2),(255,0,0),1)
          #      
          #      
    
       

   
   
    

video_capture.release()
out.release()
cv2.destroyAllWindows()



