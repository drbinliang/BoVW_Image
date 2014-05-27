'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import os

def isGray(image):
    """ Return True if the image has one channel per pixel. """
    return image.ndim < 3

def write2SVMFormat(outputPath, fileName, X, y):
        ''' write X, y to SVM format 
            y: list of labels [1, -1, 1]
            X: list of data [[1,1,1], [1,2,3], ...]
        '''
        # Create the output
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        
        problemFile = open(os.path.join(outputPath, fileName), 'w')
        numData = len(y)        # number of data
        dimFeature = len(X[0])  # dimensionality of feature
        
        for i in xrange(numData):
            yi = y[i]
            Xi = X[i]
            
             
            # write to SVM-Format file
            problemFile.write('%d' % yi)
            problemFile.write(' ')

            for j in xrange(dimFeature):
                data = Xi[j]

                problemFile.write('%d' % (j + 1))
                problemFile.write(':')
                problemFile.write('%f' % data)
                problemFile.write(' ')
          
            problemFile.write('\n')  
       
        problemFile.close()