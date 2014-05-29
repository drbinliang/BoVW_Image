'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import cv2
from utils import isGray
from scipy.cluster.vq import vq
import numpy as np
import config

def extractSiftFeatures(image):
    """ detect interest points in an image """
    
    if not isGray(image):
        # convert RGB image to gray image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    sift = cv2.SIFT()
    
    # kps: a list of keypoints
    # des: numpy array of shape [Number_of_Keypoints x 128]
    #      each row represents an observation
    kps, des = sift.detectAndCompute(image, None)
    
    return des

def encodeFeatures(siftFeatures, codebook):
    """ Encode features using codebook """
    index, _ = vq(siftFeatures, codebook)
    row, _ = siftFeatures.shape
    col = config.codebookSize
    encodedFeatures = np.zeros((row, col))
    
    for i in xrange(len(index)):
        encodedFeatures[i, index[i]] = 1
            
    return encodedFeatures
