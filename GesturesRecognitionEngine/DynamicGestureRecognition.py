import cv2, pickle
import numpy as np
#import tensorflow as tf
#from cnn_tf import cnn_model_fn
#import os
import sqlite3
from keras.models import load_model

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.logging.set_verbosity(tf.logging.ERROR)
prediction = None
model = load_model('cnn_model_keras2.h5')

image_x, image_y = cv2.imread('gestures/0/100.jpg', 0).shape

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def split_sentence(text, num_of_words):
	'''
	Splits a text into group of num_of_words
	'''
	list_words = text.split(" ")
	length = len(list_words)
	splitted_sentence = []
	b_index = 0
	e_index = num_of_words
	while length > 0:
		part = ""
		for word in list_words[b_index:e_index]:
			part = part + " " + word
		splitted_sentence.append(part)
		b_index += num_of_words
		e_index += num_of_words
		length -= num_of_words
	return splitted_sentence

def put_splitted_text_in_blackboard(blackboard, splitted_text):
	y = 200
	for text in splitted_text:
		cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		y += 50

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def recognize():
	#global prediction
	video = cv2.VideoCapture(1)
	if video.read()[0] == False:
		video = cv2.VideoCapture(0)
	hist = get_hand_hist()
	width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
	#x, y, w, h = 300, 100, 300, 300
	faceClassifierPath = "haarcascade_frontalface_default.xml"
	fistClassifierPath = "fist.xml"
	palmClassifierPath = "palm.xml"
	faceClassifier = cv2.CascadeClassifier(faceClassifierPath)
	fistClassifier = cv2.CascadeClassifier(fistClassifierPath)
	palmClassifier = cv2.CascadeClassifier(palmClassifierPath)
	#print(2)
	while True:
		text = ""
		#image = video.read()[1]
		ret, image = video.read()
		#resizedImage = cv2.resize(image, (0,0), fx=1, fy=1)
		
    	# resizedImage = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
    	# cv2.imshow("Image" ,resizedImage)
		grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
		face = faceClassifier.detectMultiScale(grayScaleImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
		flag = 0
		
		#print(3)

		if(len(face)):
    
			for (x, y, w, h) in face:
				cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
    	        #print(x)
    	        #print(y)
				rightHandImage = image[0:height, 0:x+int((w/2))]
#				leftHandImage = image[0:height, x+int((w/2)):width]
#				flippedLeftHandImage = cv2.flip(leftHandImage, 1)
                
			##Fist and palm detection for right hand
			fist = fistClassifier.detectMultiScale(rightHandImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    		# print(len(fists))
			
			if(len(fist)):
				for (x, y, w, h) in fist:
					#print(4)
					flag = 1
					#print("x=%f, y=%f", (x,y))
					# cv2.rectangle(image, (x, y-100), (x+w+100, y+h), (0, 0, 255), 2)
					# rightx = x
					# rightw = x+w+100
					# righty = y-100
					# righth = y+h
					cv2.rectangle(image, (x-50, y-100), (x+250, y+200), (0, 0, 255), 2)
					rightx = x-50
					rightw = x+250
					righty = y-100
					righth = y+200
			else:
				palm = palmClassifier.detectMultiScale(rightHandImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
				for (x, y, w, h) in palm:
					#print(5)
					flag = 1
					#print("x=%f, y=%f", (x,y))
					cv2.rectangle(image, (x-100, y-75), (x+200, y+225), (255, 0, 255), 2)
					rightx = x-100
					rightw = x+200
					righty = y-75
					righth = y+225   
    
			##Fist and palm detection for left hand    
#			fist = fistClassifier.detectMultiScale(flippedLeftHandImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 		   
#==============================================================================
#     		# print(len(fists))
# 			if(len(fist)):
# 				for (x, y, w, h) in fist:
#     				# print("x=%f, y=%f", (x,y))
# 					cv2.rectangle(image, (width-x, y), (width-x-w, y+h), (0, 0, 255), 2)
# 			else:
# 				palm = palmClassifier.detectMultiScale(flippedLeftHandImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# 				for (x, y, w, h) in palm:
#     				# print("x=%f, y=%f", (x,y))
# 					cv2.rectangle(image, (width-x, y), (width-x-w, y+h), (255, 0, 255), 2)
#     
#==============================================================================
		#cv2.imshow("Image" ,image)
    	# cv2.imshow("Image1" ,leftHandImage)
    	# cv2.imshow("Image2" ,flippedLeftHandImage)
    
    
    	#if cv2.waitKey(1) & 0xFF == ord('q'):
    	#	break 
    
		if(flag):
			#print(7)
			#image = cv2.flip(image, 1)
			#imgCrop = image[y:y+h, x:x+w]
			#print(image.shape)  
			imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
			cv2.filter2D(dst,-1,disc,dst)
			blur = cv2.GaussianBlur(dst, (11,11), 0)
			blur = cv2.medianBlur(blur, 15)
			thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
			thresh = cv2.merge((thresh,thresh,thresh))
			thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
			thresh = thresh[righty:righth,rightx:rightw]
#			contours,tmp = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
			contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
#	
			if len(contours) > 0:
				contour = max(contours, key = cv2.contourArea)
				#print(cv2.contourArea(contour))
				if cv2.contourArea(contour) > 10000:
					x1, y1, w1, h1 = cv2.boundingRect(contour)
					save_img = thresh[y1:y1+h1, x1:x1+w1]
					
					if w1 > h1:
						save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
					elif h1 > w1:
						save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
    				
					pred_probab, pred_class = keras_predict(model, save_img)
					###print(pred_class, pred_probab)
    				
					if pred_probab > 0.8:
						text = get_pred_text_from_db(pred_class)
						###print(text)
			blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
			splitted_text = split_sentence(text, 2)
			put_splitted_text_in_blackboard(blackboard, splitted_text)
			#cv2.putText(blackboard, text, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255))
			#cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
			res = np.hstack((image, blackboard))
			cv2.imshow("Recognizing gesture", res)
			if(len(fist)):
				cv2.imshow("thresh", thresh)
			#print(8)
			if cv2.waitKey(1) == ord('q'):
				break

#keras_predict(model, np.zeros((50, 50), dtype=np.uint8))	
#print(1)
recognize()