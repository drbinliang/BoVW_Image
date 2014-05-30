'''
Created on 30/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import os
import config
from sklearn import preprocessing
from svm import svm_problem, svm_parameter
from svmutil import svm_train, svm_predict

class SvmTool(object):
    
    def __init__(self, train_y, train_X, kernel = 'RBF'):
        self._kernel = kernel
        self._scaler = None
        self.train_X = train_X
        self.train_y = train_y
        self._param_c = config.svm_c
        self._param_g = config.svm_g
        self._model = None
    
    
    def learnModel(self):
        # scale train data
        svmScaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
        train_X_scaledArr = svmScaler.fit_transform(self.train_X)
        
        # learn and save svm model
        X = train_X_scaledArr.tolist()   
        problem = svm_problem(self.train_y, X)
        paramStr = '-c ' + str(self._param_c) + ' -g ' + str(self._param_g) + ' -q'
        param = svm_parameter(paramStr)
        
        self._model = svm_train(problem, param)
        self._scaler = svmScaler
        
    
    def doPredication(self, test_y, test_X):
        """ Test data using learned svm model """
    
        # scale
        test_X_scaledArr = self._scaler.transform(test_X)
        X = test_X_scaledArr.tolist()
        
        p_label, p_acc, _ = svm_predict(test_y, X, self._model, '-q')
        
        accuracy, _, _ = p_acc
        
        return p_label, accuracy 
    
    
    def write2SVMFormat(self, outputPath, fileName, X, y):
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
        