'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import config
import os
import cv2
import numpy as np
from codebook.codebook_generatation import generateCodebook
from recognition.train_data import learnSvmModel
from recognition.test_data import classifyData
from validation.cross_validation import crossValidate
from domain.image import ImageData

def main():
    imageDatasetDir = config.imageDatasetDir
    
    # Select part of the whole categories except 'BACKGROUND_Google'
    categories = os.listdir(imageDatasetDir)
    categories = categories[:config.numCategories]
    
    allFeatures = np.array([])
    
    ## Step.1 Data loading and features extraction 
    # Get features from training data over all categories
    print "Data loading and feature extraction ..."
    
    trainImageData = []
    testImageData = []
    
    for category in categories:
        categoryPath = os.path.join(imageDatasetDir, category)
        allData = os.listdir(categoryPath)
        numData = len(allData) # number of all data of the category
        numTrainData = int(numData * config.percentageTrainData)    # number of training data
        
        trainData = allData[:numTrainData]
        testData = allData[numTrainData:]
        
        # Train data loading
        for data in trainData:
            filePath = os.path.join(categoryPath, data)
            
            imageData = ImageData(filePath)
            imageData.extractFeatures()
            imageData.className = category
            imageData.classId = categories.index(category)
            
            trainImageData.append(imageData)
            
            if allFeatures.size == 0:
                allFeatures = imageData.features
            else:
                allFeatures = np.vstack((allFeatures, imageData.features))
        
        # Test data loading
        for data in testData:
            filePath = os.path.join(categoryPath, data)
            
            imageData = ImageData(filePath)
            imageData.extractFeatures()
            imageData.className = category
            imageData.classId = categories.index(category)
            
            testImageData.append(imageData)
    
    ## Step.2 Codebook generation
    print "Codebook generation ..."
    codebook = generateCodebook(allFeatures)
    
    ## Step.3 Feature encoding for train data
    train_y = []
    train_X = []
    
    for imageData in trainImageData:
        imageData.encodeFeatures(codebook)
        
        # Format train data
        train_y.append(imageData.classId)
        train_X.append(imageData.encodedFeatures)
    
    # Cross validation    
    if config.is_cv:
        # cross validation
        crossValidate(train_y, train_X)
    
    ## Step.4 Classification
    # Learning using SVM
    print "Model learning ..."
    svmModel, svmScaler = learnSvmModel(train_y, train_X, config.svm_c, config.svm_g)
    
    # Feature encoding for test data and classify data using learned model
    print "Classifying ..."
    numCorrect = 0
    for imageData in testImageData:
        imageData.encodeFeatures(codebook)
        
        # Format train data
        test_y = [imageData.classId]
        test_X = [imageData.encodedFeatures]
        
        p_label, _ = classifyData(test_y, test_X, svmModel, svmScaler)
        predClassId = int(p_label[0])
        predClassName = categories[predClassId]
        
        # Write test image to predicated category
        predClassPath = os.path.join(config.predctDir, predClassName)
        if not os.path.exists(predClassPath):
            os.makedirs(predClassPath)
        
        imageName = imageData.className + '_' + os.path.basename(imageData.filepath)
        predFilePath = os.path.join(predClassPath, imageName)
        
        cv2.imwrite(predFilePath, imageData.image)
        
        if predClassId == imageData.classId:
            numCorrect += 1 
    
    # Calculate results
    accuracy = numCorrect / float(len(testImageData))
    print "accuracy: ", accuracy

if __name__ == '__main__':
    main()