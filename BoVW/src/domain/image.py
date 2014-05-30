'''
Created on 28/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import cv2
from extraction.feature_extraction import extractSiftFeatures
import numpy as np

class ImageData(object):
    
    def __init__(self, filePath):
        self.filepath = filePath
        self.image = cv2.imread(filePath)
        
        self.className = None
        self.classId = -1
        self._features = None
        self._encodedFeatures = None
        self._pooledFeatures = None
        self._finalFeatures = None
    
    
    @property
    def features(self):
        return self._features
    
    
    @property
    def finalFeatures(self):
        return self._finalFeatures
    
        
    def extractFeatures(self):
        """ Extract _features """
        self._features = extractSiftFeatures(self.image)
     
     
    def generateFinalFeatures(self, bovw):
        """ Generate final _features using bag of visual words (bovw) """
        # 1. Do feature encoding
        self._encodedFeatures = bovw.doFeatureEncoding(self._features)
        
        # 2. Do feature pooling
        self._pooledFeatures = bovw.doFeaturePooling(self._encodedFeatures)
        
        # 3. Do normalization
        self._finalFeatures = bovw.doFeatureNormalization(self._pooledFeatures)
        
        
    def poolFeature(self):
        """ feature pooling """
        # sum pooling
        self.pooledFeatures = np.average(self.encodedFeatures, axis = 0)
        