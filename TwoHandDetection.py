import cv2
import numpy as np
import time
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
#log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# When everything is done, release the capture
#video_capture.release()
#cv2.destroyAllWindows()

#Open Camera object
#cap = cv2.VideoCapture(0)

#Decrease frame size
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

def nothing(x):
    pass

# Function to find angle between two vectors
def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

# Function to find distance between two points in a list of lists
def FindDistance(A,B): 
 return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2)) 
 

# Creating a window for HSV track bars
#cv2.namedWindow('HSV_TrackBar')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
"""###
cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)
###"""

while(1):
    
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        left_image = frame[0:height, 0:x+int((w/2))]
        #cv2.imwrite("left_image.png",img_left)
        right_image = frame[0:height, x+int((w/2)):width]
        #cv2.imwrite("right_image.png",img_right)
        

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

#while(1):


    
    #Kernel matrices for morphological transformation    
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    #Measure execution time 
    start_time = time.time()
    
    #Capture frames from the camera
    frame_left = left_image
    
    #Blur the image
    blur_left = cv2.blur(frame_left,(3,3))
 	
    #Convert to HSV color space
    hsv_left = cv2.cvtColor(blur_left,cv2.COLOR_BGR2HSV)
    
    #Create a binary image with where white will be skin colors and rest is black
    mask2_left = cv2.inRange(hsv_left,np.array([2,50,50]),np.array([15,255,255]))
    #cv2.imshow("thresholded", mask2)
    
    
    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation_left = cv2.dilate(mask2_left,kernel_ellipse,iterations = 1)
    erosion_left = cv2.erode(dilation_left,kernel_square,iterations = 1)    
    dilation2_left = cv2.dilate(erosion_left,kernel_ellipse,iterations = 1)    
    filtered_left = cv2.medianBlur(dilation2_left,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2_left = cv2.dilate(filtered_left,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3_left = cv2.dilate(filtered_left,kernel_ellipse,iterations = 1)
    median_left = cv2.medianBlur(dilation2_left,5)
    ret,thresh_left = cv2.threshold(median_left,127,255,0)
    
    #Find contours of the filtered frame
    contours_left, hierarchy = cv2.findContours(thresh_left,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
    #Draw Contours
    #cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
    #cv2.imshow('Dilation',median)
    
	#Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0
    	
    for i in range(len(contours_left)):
        cnt=contours_left[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i  
            
	#Largest area contour
	
    if(ci < len(contours_left)):		 			  
        cnts = contours_left[ci]
    	
    #Find convex hull
    hull = cv2.convexHull(cnts)
    
    #Find convex defects
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)
    
    #Get defect points and draw them in the original image
    
    
    #Print number of pointed fingers
    ###cv2.putText(frame,str(result),(100,100),font,2,(255,255,255),2)
    
    #show height raised fingers
    #cv2.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)
        
    #Print bounding rectangle
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cropped_frame = img[y:y+h, x:x+w]
    
    #Blur the image
    blur = cv2.blur(cropped_frame,(3,3))
    
    #Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
    
    resize_mask2_left = cv2.resize(mask2, (300,300))
    cv2.imshow("thresholded_left", resize_mask2_left)
    #fx = 500/w
    #fy = 500/h
    cv2.resizeWindow("thresholded_left",300,300)

    
        
    
    
    
    
    
    
    
    
    frame_right = right_image
    blur_right = cv2.blur(frame_right,(3,3))
    hsv_right = cv2.cvtColor(blur_right,cv2.COLOR_BGR2HSV)
    mask2_right = cv2.inRange(hsv_right,np.array([2,50,50]),np.array([15,255,255]))
    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation_right = cv2.dilate(mask2_right,kernel_ellipse,iterations = 1)
    erosion_right = cv2.erode(dilation_right,kernel_square,iterations = 1)    
    dilation2_right = cv2.dilate(erosion_right,kernel_ellipse,iterations = 1)    
    filtered_right = cv2.medianBlur(dilation2_right,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2_right = cv2.dilate(filtered_right,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3_right = cv2.dilate(filtered_right,kernel_ellipse,iterations = 1)
    median_right = cv2.medianBlur(dilation2_right,5)
    ret,thresh_right = cv2.threshold(median_right,127,255,0)
    
    #Find contours of the filtered frame
    contours_right, hierarchy = cv2.findContours(thresh_right,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
    #Draw Contours
    #cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
    #cv2.imshow('Dilation',median)
    
	#Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0
    	
    for i in range(len(contours_right)):
        cnt=contours_right[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i  
            
	#Largest area contour
	
    if(ci < len(contours_right)):		 			  
        cnts = contours_right[ci]
    	
    #Find convex hull
    hull = cv2.convexHull(cnts)
    
    #Find convex defects
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)
    
    #Get defect points and draw them in the original image
    
    
    #Print number of pointed fingers
    ###cv2.putText(frame,str(result),(100,100),font,2,(255,255,255),2)
    
    #show height raised fingers
    #cv2.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
    #cv2.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)
        
    #Print bounding rectangle
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cropped_frame = img[y:y+h, x:x+w]
    cv2.imwrite("rightHand.png",cropped_frame)
    
    #Blur the image
    blur = cv2.blur(cropped_frame,(3,3))
    
    #Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
    
    resize_mask2_right = cv2.resize(mask2, (300,300))
    cv2.imshow("thresholded_right", resize_mask2_right)
    #fx = 500/w
    #fy = 500/h
    cv2.resizeWindow("thresholded_right",300,300)
    
    #cv2.drawContours(frame,[hull],-1,(255,255,255),2)
    
    ##### Show final image ########
    #cv2.imshow('Hand detection',frame)
    ###############################
    
    #Print execution time
    #print time.time()-start_time
    
    #close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
