'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import config
import os
import cv2
from extraction.feature_extraction import extractSiftFeatures, encodeFeatures
import numpy as np
from codebook.codebook_generatation import generateCodebook
from recognition.train_data import learnSvmModel
from recognition.test_data import classifyData
from validation.cross_validation import crossValidate
import matplotlib.pyplot as plt

def main():
    imageDatasetFolder = config.imageDatasetFolder
    categories = os.listdir(imageDatasetFolder)[:config.numCategories]
    
    allFeatures = np.array([])
    
    # Get all features
    for category in categories:
        categoryPath = os.path.join(imageDatasetFolder, category)
        allData = os.listdir(categoryPath)
        numData = len(allData) # number of whole data of the category
        numTrainData = int(numData * config.percentageTrainData)    # number of training data
        trainData = allData[:numTrainData]
        
        for data in trainData:
            filePath = os.path.join(categoryPath, data)
            image = cv2.imread(filePath)
            siftFeatures = extractSiftFeatures(image)
            
            if allFeatures.size == 0:
                allFeatures = siftFeatures
            else:
                allFeatures = np.vstack((allFeatures, siftFeatures))
    
    # Generate codebook
    codebook = generateCodebook(allFeatures)
    
    # Feature encoding
    train_y = []
    train_X = []
    test_y = []
    test_X = []
    for category in categories:
        categoryPath = os.path.join(imageDatasetFolder, category)
        allData = os.listdir(categoryPath)
        numData = len(allData) # number of whole data of the category
        numTrainData = int(numData * config.percentageTrainData)    # number of training data
        trainData = allData[:numTrainData]
        testData = allData[numTrainData:]
        
        # training data processing
        for data in trainData:
            filePath = os.path.join(categoryPath, data)
            image = cv2.imread(filePath)
            siftFeatures = extractSiftFeatures(image)
            
            label = categories.index(category)
            
            # Feature encoding
            encodedFeatures = encodeFeatures(siftFeatures, codebook)
            
            train_y.append(label)
            train_X.append(encodedFeatures)
            
        # test data processing
        for data in testData:
            filePath = os.path.join(categoryPath, data)
            image = cv2.imread(filePath)
            siftFeatures = extractSiftFeatures(image)
            
            label = categories.index(category)
            
            # Feature encoding
            encodedFeatures = encodeFeatures(siftFeatures, codebook)
            
            test_y.append(label)
            test_X.append(encodedFeatures)
    
    if config.is_cv:
        # cross validation
        crossValidate(train_y, train_X)
    
    # Learning using SVM
    svmModel, svmScaler = learnSvmModel(train_y, train_X, config.svm_c, config.svm_g)
    
    # test data using learned model
    p_label, accuracy = classifyData(test_y, test_X, svmModel, svmScaler)
    
    # print result
    print "accuracy: ", accuracy

if __name__ == '__main__':
    main()