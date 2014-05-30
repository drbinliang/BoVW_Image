'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import cv2
from utils import isGray

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
