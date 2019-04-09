import cv2

faceClassifierPath = "haarcascade_frontalface_default.xml"
fistClassifierPath = "fist.xml"
palmClassifierPath = "palm.xml"
faceClassifier = cv2.CascadeClassifier(faceClassifierPath)
fistClassifier = cv2.CascadeClassifier(fistClassifierPath)
palmClassifier = cv2.CascadeClassifier(palmClassifierPath)

# image = cv2.imread("frame-1.jpg")
video = cv2.VideoCapture(0)


while True:
	ret, image = video.read()
	resizedImage = cv2.resize(image, (0,0), fx=1, fy=1)
	# resizedImage = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
	# cv2.imshow("Image" ,resizedImage)
	grayScaleImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)

	face = faceClassifier.detectMultiScale(grayScaleImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

	if(len(face)):

		for (x, y, w, h) in face:
			cv2.rectangle(resizedImage, (x, y), (x+w, y+h), (0, 0, 0), 2)
	        #print(x)
	        #print(y)
			width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

			rightHandImage = image[0:height, 0:x+int((w/2))]
			leftHandImage = image[0:height, x+int((w/2)):width]
			midFaceCoordinate = x+int((w/2))
			flippedLeftHandImage = cv2.flip(leftHandImage, 1)


		fist = fistClassifier.detectMultiScale(rightHandImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

		# print(len(fists))
		if(len(fist)):
			for (x, y, w, h) in fist:
				# print("x=%f, y=%f", (x,y))
				cv2.rectangle(resizedImage, (x, y), (x+w, y+h), (0, 0, 255), 2)
		else:
			palm = palmClassifier.detectMultiScale(rightHandImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
			for (x, y, w, h) in palm:
				# print("x=%f, y=%f", (x,y))
				cv2.rectangle(resizedImage, (x, y), (x+w, y+h), (255, 0, 255), 2)


		fist = fistClassifier.detectMultiScale(flippedLeftHandImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

		# print(len(fists))
		if(len(fist)):
			for (x, y, w, h) in fist:
				# print("x=%f, y=%f", (x,y))
				cv2.rectangle(resizedImage, (width-x, y), (width-x-w, y+h), (0, 0, 255), 2)
		else:
			palm = palmClassifier.detectMultiScale(flippedLeftHandImage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
			for (x, y, w, h) in palm:
				# print("x=%f, y=%f", (x,y))
				cv2.rectangle(resizedImage, (width-x, y), (width-x-w, y+h), (255, 0, 255), 2)

	cv2.imshow("Image" ,resizedImage)
	# cv2.imshow("Image1" ,leftHandImage)
	# cv2.imshow("Image2" ,flippedLeftHandImage)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break