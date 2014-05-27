'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
from sklearn import preprocessing
from svm import svm_problem, svm_parameter
from svmutil import svm_train

def learnSvmModel(train_y, train_X, svm_c, svm_g):
    """ learn svm model """
    
    # scale train data
    svmScaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    train_X_scaledArr = svmScaler.fit_transform(train_X)
    
    # learn and save svm model
    X = train_X_scaledArr.tolist()   
    problem = svm_problem(train_y, X)
    paramStr = '-c ' + str(svm_c) + ' -g ' + str(svm_g)
    param = svm_parameter(paramStr)
    svmModel = svm_train(problem, param)
    
    return svmModel, svmScaler