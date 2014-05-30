'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
from sklearn import preprocessing
from recognition.svm_tool import SvmTool
def crossValidate(train_y, train_X):
    """ Cross validate to get optimal parameters """
    # scale data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_scaledArr = min_max_scaler.fit_transform(train_X)
    X_scaled = X_scaledArr.tolist()
    
    # write to svm format file
    outputPath = '.\\cv'
    fileName = 'train_data'
    svmTool = SvmTool()
    svmTool.write2SVMFormat(outputPath, fileName, X_scaled, train_y)