'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
from scipy.cluster.vq import whiten, kmeans2, ClusterError
import config
import numpy as np
import os

def generateCodebook(features):
    """ Generate codebook using extracted features """
    
    codebook = None
    
    if os.path.exists('codebook.npy'):
        codebook = np.load('codebook.npy')
    else:
        whitenedFeatures = whiten(features)
        run = True
        
        while run:
            try:
                # Set missing = 'raise' to raise exception 
                # when one of the clusters is empty
                codebook, _ = kmeans2(whitenedFeatures, 
                                      config.codebookSize, 
                                      missing = 'raise')
                
                # No empty clusters
                run = False
            except ClusterError:
                # If one the clusters is empty, re-run k-means
                run = True
    
        # Save codebook to disk
        np.save('codebook', codebook)
    
    return codebook