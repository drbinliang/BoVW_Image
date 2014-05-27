'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
from scipy.cluster.vq import whiten, kmeans2
import config

def generateCodebook(features):
    """ Generate codebook using extracted features """
    whitenedFeatures = whiten(features)
    
    codebook, _ = kmeans2(whitenedFeatures, config.codebookSize)
    
    
    return codebook