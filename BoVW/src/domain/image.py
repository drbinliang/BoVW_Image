'''
Created on 28/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import cv2
from extraction.feature_extraction import extractSiftFeatures, encodeFeatures

class ImageData(object):
    
    def __init__(self, filePath):
        self.filepath = filePath
        self.image = cv2.imread(filePath)
        
        self.className = None
        self.classId = -1
        self.features = None
        self.encodedFeatures = None
        
    def extractFeatures(self):
        """ Extract features """
        self.features = extractSiftFeatures(self.image)
        
    def encodeFeatures(self, codebook):
        """ Get encoded features using codebook """
        self.encodedFeatures = encodeFeatures(self.features, codebook)