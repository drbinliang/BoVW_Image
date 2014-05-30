'''
Created on 29/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
from scipy.cluster.vq import whiten, kmeans2, ClusterError, vq
import config
import numpy as np
from numpy import linalg as LA

class BagOfWords(object):
    
    def __init__(self, 
                 codebookGenerateMethod = 'k-means', 
                 featureEncodingMethod = 'hard-assignment',
                 poolingMethod = 'sum-pooling',
                 normalizationMethod = 'L1-norm'):
        
        self._codebookSize = config.codebookSize    # codebook size
        self._codebook = None
        self._codebookGenerateMethod = codebookGenerateMethod
        self._featureEncodingMethod = featureEncodingMethod
        self._poolingMethod = poolingMethod
        self._normalizationMethod = normalizationMethod
    
    @property
    def codebook(self):
        return self._codebook
    
    def generateCodebook(self, features):
        """ Generate codebook using extracted features """
    
        whitenedFeatures = whiten(features)
        run = True
        codebook = None
        
        if self._codebookGenerateMethod == 'k-means':
            # Codebook generation using k-means
            while run:
                try:
                    # Set missing = 'raise' to raise exception 
                    # when one of the clusters is empty
                    codebook, _ = kmeans2(whitenedFeatures, 
                                          self._codebookSize, 
                                          missing = 'raise')
                    
                    # No empty clusters
                    run = False
                except ClusterError:
                    # If one of the clusters is empty, re-run k-means
                    run = True
        else:
            pass
        
        self._codebook = codebook
        
    
    def doFeatureEncoding(self, features):
        """ do feature encoding to original features"""
        encodedFeatures = None
        
        if self._featureEncodingMethod == 'hard-assignment':
            # Hard assignment
            index, _ = vq(features, self._codebook)
            row, _ = features.shape
            col = config.codebookSize
            encodedFeatures = np.zeros((row, col))
            
            for i in xrange(len(index)):
                encodedFeatures[i, index[i]] = 1
        else:
            pass
                
        return encodedFeatures
    
    
    def doFeaturePooling(self, encodedFeatures):
        """ Do feature pooling to encoded features """
        pooledFeatures = None
        if self._poolingMethod == 'sum-pooling':
            # Sum pooling
            pooledFeatures = np.sum(encodedFeatures, axis = 0)
        else:
            pass
        
        return pooledFeatures
    
    
    def doFeatureNormalization(self, pooledFeatures):
        """ Do feature normalization to pooled features """
        normFeatures = None
        if self._normalizationMethod == 'L1-norm':
            # L1-normalization
            normFeatures = pooledFeatures / LA.norm(pooledFeatures, ord = 1)
        else:
            pass
        
        return normFeatures