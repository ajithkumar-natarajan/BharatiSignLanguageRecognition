import cv2
import numpy as np
import pickle
from PIL import Image
import keras
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#imag = cv2.imread("100.png",0)
#for i in range(3):
#i = 3
#img = cv2.imread(str(i+1)+".jpg")
#for j in range(600):
#    val = (i+1)+(j*4)
#    cv2.imwrite(str(val)+".jpg",img)


#for i in range(9000,9999):
#    num = i+2
#    img = Image.open("Hand_"+"000"+str(num)+".jpg")
#    img = img.resize((50,50))
#    img.save(str(num)+".jpg")
#    
#a=np.zeros([1,1,1])
fin=[]
for po in range(9):
    fname = "pos"+str(po+1)+".jpg"
    img= cv2.imread(fname,0)
#    R=img[:,:,1]
#img.resize((400,520))
#keras.layers.Conv2D(filters=(20,20), kernel_size=(3,3), strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
#keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
    aa=skimage.measure.block_reduce(img, (3,3), np.max)
    fin.append(aa)
    cv2.imshow('image'+str(po),aa)
    cv2.waitKey(0)

