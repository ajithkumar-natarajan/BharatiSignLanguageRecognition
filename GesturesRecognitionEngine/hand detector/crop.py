# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:23:54 2019

@author: Sreya
"""
import cv2
import numpy as np

#val=0
#image=cv2.imread('ROI.jpg',0)
#dim=image.shape[0:2]
#kernelx = np.array([[-1,1,2]])
#
#for i in range(dim[0]):
#    hval=0
#    for j in range(dim[1]):
#        pix=image[i,j]
#        comp=np.multiply(kernelx,pix)
#        if hval==0:
#            hout=comp
#            hval=1
#        else:
#            hout=np.append(hout,comp,1)
#    if val==0:
#        out=hout
#        val=1
#    else:
#        out=np.append(out,hout,0)
#        
#out[out<120]=0
#out[out>180]=0
#out=np.uint8(out)
#ret,thresh=cv2.threshold(out,140,255,cv2.THRESH_BINARY)
#cv2.imshow("out",thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

l=0,u=3
fin=np.add(out[:,0],out[:,1],out[:,2])
