##Calibrate and capture train data and train MLP
import keras
import pickle
import cv2
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score
import skimage.measure
s = tf.InteractiveSession()

def build_squares(img):
	x, y, w, h = 420, 140, 10, 10
	d = 10
	imgCrop = None
	crop = None
	for i in range(10):
		for j in range(5):
			if np.any(imgCrop == None):
				imgCrop = img[y:y+h, x:x+w]
			else:
				imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
			#print(imgCrop.shape)
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
			x+=w+d
		if np.any(crop == None):
			crop = imgCrop
		else:
			crop = np.vstack((crop, imgCrop)) 
		imgCrop = None
		x = 420
		y+=h+d
	return crop
8
cam = cv2.VideoCapture(0)
x, y, w, h = 300, 100, 300, 300
flagPressedC, flagPressedS = False, False
imgCrop = None
while cam.read()[0]==True:
    img = cam.read()[1]
    img = cv2.flip(img, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    key = cv2.waitKey(1)
    if key == ord('c'):
        hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        flagPressedC = True
        #    cv2.imshow('thresh',img)
    if flagPressedC:
        dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
        dst1 = dst.copy()
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        cv2.filter2D(dst,-1,disc,dst)  
        blur = cv2.GaussianBlur(dst, (11,11), 0)
        blur = cv2.medianBlur(blur, 15)
        ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.merge((thresh,thresh,thresh))
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(thresh,(x,y),(x+w,y+h),(255,255,255),2)
            thresh[x:x+w,y:y+h]=0
        #cv2.imshow("res", res)
        cv2.imshow("Thresh", thresh)
    if not flagPressedS:
        imgCrop = build_squares(img)
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Set hand histogram", img)
    if key==ord('1'):
        cv2.imwrite('pos1.jpg',thresh)
    elif key==ord('2'):
        cv2.imwrite('pos2.jpg',thresh)
    elif key==ord('3'):
        cv2.imwrite('pos3.jpg',thresh)
    elif key==ord('4'):
        cv2.imwrite('pos4.jpg',thresh)
    elif key==ord('5'):
        cv2.imwrite('pos5.jpg',thresh)
    elif key==ord('6'):
        cv2.imwrite('pos6.jpg',thresh)
    elif key==ord('7'):
        cv2.imwrite('pos7.jpg',thresh)
    elif key==ord('8'):
        cv2.imwrite('pos8.jpg',thresh)
    elif key==ord('9'):
        cv2.imwrite('pos9.jpg',thresh)
    elif key==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

def load_dataset(flatten=False):
    cla=np.zeros([900,9])
    all = np.zeros([900,160,214])
    l_r,u_r=0,99
    for po in range(1,10):
        fname="pos"+str(po)+".jpg"
        img= cv2.imread(fname,0)
        img=skimage.measure.block_reduce(img, (3,3), np.max)
#        img.resize((240,320))
        all[l_r:u_r,:,:]=img
        cla[l_r:u_r,po-1]=1
        l_r,u_r=l_r+100,u_r+100
    X_train=all
    y_train=cla
    X_test=X_train[23:43,:,:]
    y_test=y_train[23:43,:]
    return X_train, y_train, X_test,y_test

X_train, y_train, X_test, y_test = load_dataset()

## Changing dimension of input images from N*28*28 to  N*784
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))

print('Train dimension:');print(X_train.shape)
print('Test dimension:');print(X_test.shape)

## Defining various initialization parameters for 784-512-256-10 MLP model
num_classes = y_train.shape[1]
num_features = X_train.shape[1]
num_output = y_train.shape[1]
num_layers_0 = 2500
num_layers_1 = 250
num_layers_2 = 25
starter_learning_rate = 0.01
regularizer_rate = 0.1

# Placeholders for the input data
input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')

## for dropout layer
keep_prob = tf.placeholder(tf.float32)

## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
weights_0 = tf.Variable(tf.random_normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
bias_0 = tf.Variable(tf.random_normal([num_layers_0]))

weights_1 = tf.Variable(tf.random_normal([num_layers_0,num_layers_1], stddev=(1/tf.sqrt(float(num_layers_0)))))
bias_1 = tf.Variable(tf.random_normal([num_layers_1]))

weights_2 = tf.Variable(tf.random_normal([num_layers_1,num_layers_2], stddev=(1/tf.sqrt(float(num_layers_1)))))
bias_2 = tf.Variable(tf.random_normal([num_layers_2]))

weights_3 = tf.Variable(tf.random_normal([num_layers_2,num_output], stddev=(1/tf.sqrt(float(num_layers_2)))))
bias_3 = tf.Variable(tf.random_normal([num_output]))

## Initializing weigths and biases
hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)

hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0,weights_1)+bias_1)
hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)

hidden_output_2 = tf.nn.relu(tf.matmul(hidden_output_1_1,weights_2)+bias_2)
hidden_output_2_2 = tf.nn.dropout(hidden_output_2, keep_prob)

predicted_y = tf.sigmoid(tf.matmul(hidden_output_2_2,weights_3) + bias_3)

## Defining the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y,labels=input_y)) \
        + regularizer_rate*(tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))

## Variable learning rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)

## Adam optimzer for finding the right weight
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,weights_2,bias_0,bias_1,bias_2])
    
## Metrics definition
correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predicted_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## Training parameters
batch_size = 10
epochs=3
dropout_prob = 0.6

training_accuracy = []
training_loss = []
testing_accuracy = []

s.run(tf.global_variables_initializer())
for epoch in range(epochs):    
    arr = np.arange(X_train.shape[0])
    np.random.shuffle(arr)
    for index in range(0,X_train.shape[0],batch_size):
        s.run(optimizer, {input_X: X_train[arr[index:index+batch_size]],
                          input_y: y_train[arr[index:index+batch_size]],
                        keep_prob:dropout_prob})
        print(index)
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:X_train, 
                                                         input_y: y_train,keep_prob:1}))
    training_loss.append(s.run(loss, {input_X: X_train, 
                                      input_y: y_train,keep_prob:1}))
    
    # Evaluation of model
    testing_accuracy.append(accuracy_score(y_test.argmax(1), 
                            s.run(predicted_y, {input_X: X_test,keep_prob:1}).argmax(1)))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,
                                                                    training_loss[epoch],
                                                                    training_accuracy[epoch],
                                                                   testing_accuracy[epoch]))

    
    ## Plotting chart of training and testing accuracy as a function of iterations
iterations = list(range(epochs))
plt.plot(iterations, training_accuracy, label='Train')
plt.plot(iterations, testing_accuracy, label='Test')
plt.ylabel('Accuracy')
plt.xlabel('iterations')
plt.show()
print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
print("Test Accuracy:{0:.2f}".format(testing_accuracy[-1]))