import cv2
import numpy as np
import time

#Open Camera object
cap = cv2.VideoCapture(0)

#Decrease frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

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

    #Measure execution time 
    start_time = time.time()
    
    #Capture frames from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    #Blur the image
    blur = cv2.blur(frame,(3,3))
 	
 	#Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
    #cv2.imshow("thresholded", mask2)
    
    #Kernel matrices for morphological transformation    
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,127,255,0)
    
    #Find contours of the filtered frame
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
    #Draw Contours
    #cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
    #cv2.imshow('Dilation',median)
    
	#Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0
    	
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i  
            
	#Largest area contour
	
    if(ci < len(contours)):		 			  
        cnts = contours[ci]
    	
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
    
    resize_mask2 = cv2.resize(mask2, (500,500))
    cv2.imshow("thresholded", resize_mask2)
    #fx = 500/w
    #fy = 500/h
    cv2.resizeWindow("thresholded",500,500)
    
    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
    
    ##### Show final image ########
    cv2.imshow('Hand detection',frame)
    ###############################
    
    #Print execution time
    #print time.time()-start_time
    
    #close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
