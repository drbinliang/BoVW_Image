'''
Created on 28/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import cv2
from extraction.feature_extraction import extractSiftFeatures, encodeFeatures
import numpy as np
from numpy import linalg as LA

class ImageData(object):
    
    def __init__(self, filePath):
        self.filepath = filePath
        self.image = cv2.imread(filePath)
        
        self.className = None
        self.classId = -1
        self.features = None
        self._encodedFeatures = None
        self._pooledFeatures = None
        self.finalFeatures = None
        
    def extractFeatures(self):
        """ Extract features """
        self.features = extractSiftFeatures(self.image)
     
    def generateFinalFeatures(self, codebook):
        """ Generate final features using codebook """
        # 1. Feature encoding
        self._encodedFeatures = encodeFeatures(self.features, codebook)
        
        # 2. Pooling (sum pooling)
        self._pooledFeatures = np.sum(self._encodedFeatures, axis = 0)
        
        # 3. Normalization
        self.finalFeatures = \
            self._pooledFeatures / LA.norm(self._pooledFeatures, ord = 1)
        
        
    def poolFeature(self):
        """ feature pooling """
        # sum pooling
        self.pooledFeatures = np.average(self.encodedFeatures, axis = 0)
        