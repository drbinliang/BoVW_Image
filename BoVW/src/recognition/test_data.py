'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
from svmutil import svm_predict

def classifyData(test_y, test_X, svmModel, svmScaler):
    """ Test data using learned svm model """
    
    # scale
    test_X_scaledArr = svmScaler.transform(test_X)
    X = test_X_scaledArr.tolist()
    
    p_label, p_acc, _ = svm_predict(test_y, X, svmModel, '-q')
    
    accuracy, _, _ = p_acc
    
    return p_label, accuracy