# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:51:37 2019

@author: harshith srinivas
"""

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

video_capture = cv2.VideoCapture(1)
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
    
    #left_image[] = None
    #right_image[] = None

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
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
    #cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    #cv2.imshow('Video', frame)

#while(1):


    
    #Kernel matrices for morphological transformation    
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    #Measure execution time 
    start_time = time.time()
    
    try:
        
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
        """
        #Get defect points and draw them in the original image
        FarDefect = []
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(frame_left,start,end,[0,255,0],1)
            cv2.circle(frame_left,far,10,[100,255,255],3)
        
    	#Find moments of the largest contour
        moments = cv2.moments(cnts)
        
        #Central mass of first order moments
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00
        centerMass=(cx,cy)    
        
        #Draw center mass
        cv2.circle(frame_left,centerMass,7,[100,0,255],2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_left,'Center',tuple(centerMass),font,2,(255,255,255),2)     
        
        #Distance from each finger defect(finger webbing) to the center mass
        distanceBetweenDefectsToCenter = []
        for i in range(0,len(FarDefect)):
            x =  np.array(FarDefect[i])
            centerMass = np.array(centerMass)
            distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
            distanceBetweenDefectsToCenter.append(distance)
        
        #Get an average of three shortest distances from finger webbing to center mass
        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
     
        #Get fingertip points from contour hull
        #If points are in proximity of 80 pixels, consider as a single point in the group
        finger = []
        for i in range(0,len(hull)-1):
            if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                if hull[i][0][1] <500 :
                    finger.append(hull[i][0])
        
        #The fingertip points are 5 hull points with largest y coordinates  
        finger =  sorted(finger,key=lambda x: x[1])   
        fingers = finger[0:5]
        
        #Calculate distance of each finger tip to the center mass
        fingerDistance = []
        for i in range(0,len(fingers)):
            distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
            fingerDistance.append(distance)
        
        #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        #than the distance of average finger webbing to center mass by 130 pixels
        result = 0
        for i in range(0,len(fingers)):
            if fingerDistance[i] > AverageDefectDistance+130:
                result = result +1
        
        #Print number of pointed fingers
        ###cv2.putText(frame_left,str(result),(100,100),font,2,(255,255,255),2)
        
        #show height raised fingers
        #cv2.putText(frame_left,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
        #cv2.putText(frame_left,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
        #cv2.putText(frame_left,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
        #cv2.putText(frame_left,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
        #cv2.putText(frame_left,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
        #cv2.putText(frame_left,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
        #cv2.putText(frame_left,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
        #cv2.putText(frame_left,'finger8',tuple(finger[7]),font,2,(255,255,255),2)
        """   
        #Print bounding rectangle
        x,y,w,h = cv2.boundingRect(cnts)
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #img = cv2.rectangle(frame_left,(x,y),(x+w,y+h),(0,255,0),2)
    
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #Capture frames from the camera
        frame_right = right_image
        
        #Blur the image
        blur_right = cv2.blur(frame_right,(3,3))
     	
        #Convert to HSV color space
        hsv_right = cv2.cvtColor(blur_right,cv2.COLOR_BGR2HSV)
        
        #Create a binary image with where white will be skin colors and rest is black
        mask2_right = cv2.inRange(hsv_right,np.array([2,50,50]),np.array([15,255,255]))
        #cv2.imshow("thresholded", mask2)
        
        
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
        """
        #Get defect points and draw them in the original image
        FarDefect = []
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(frame_right,start,end,[0,255,0],1)
            cv2.circle(frame_right,far,10,[100,255,255],3)
        
    	#Find moments of the largest contour
        moments = cv2.moments(cnts)
        
        #Central mass of first order moments
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00
        centerMass=(cx,cy)    
        
        #Draw center mass
        cv2.circle(frame_right,centerMass,7,[100,0,255],2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_right,'Center',tuple(centerMass),font,2,(255,255,255),2)     
        
        #Distance from each finger defect(finger webbing) to the center mass
        distanceBetweenDefectsToCenter = []
        for i in range(0,len(FarDefect)):
            x =  np.array(FarDefect[i])
            centerMass = np.array(centerMass)
            distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
            distanceBetweenDefectsToCenter.append(distance)
        
        #Get an average of three shortest distances from finger webbing to center mass
        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
     
        #Get fingertip points from contour hull
        #If points are in proximity of 80 pixels, consider as a single point in the group
        finger = []
        for i in range(0,len(hull)-1):
            if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                if hull[i][0][1] <500 :
                    finger.append(hull[i][0])
        
        #The fingertip points are 5 hull points with largest y coordinates  
        finger =  sorted(finger,key=lambda x: x[1])   
        fingers = finger[0:5]
        
        #Calculate distance of each finger tip to the center mass
        fingerDistance = []
        for i in range(0,len(fingers)):
            distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
            fingerDistance.append(distance)
        
        #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        #than the distance of average finger webbing to center mass by 130 pixels
        result = 0
        for i in range(0,len(fingers)):
            if fingerDistance[i] > AverageDefectDistance+130:
                result = result +1
        
        #Print number of pointed fingers
        ###cv2.putText(frame_right,str(result),(100,100),font,2,(255,255,255),2)
        
        #show height raised fingers
        #cv2.putText(frame_right,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
        #cv2.putText(frame_right,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
        #cv2.putText(frame_right,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
        #cv2.putText(frame_right,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
        #cv2.putText(frame_right,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
        #cv2.putText(frame_right,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
        #cv2.putText(frame_right,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
        #cv2.putText(frame_right,'finger8',tuple(finger[7]),font,2,(255,255,255),2)
        """    
        #Print bounding rectangle
        x,y,w,h = cv2.boundingRect(cnts)
        #img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        img = cv2.rectangle(frame_right,(x,y),(x+w,y+h),(0,255,0),2)
    
        cropped_frame = img[y:y+h, x:x+w]
        
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
    
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
    except NameError:
        print("No face found")
        
