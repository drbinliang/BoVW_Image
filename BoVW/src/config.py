'''
Created on 27/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''

imageDatasetDir = '.\\image_dataset'
predctDir = '.\\pred'

percentageTrainData = 0.7   # the percentage of training data
numCategories = 10   # the number of categories

codebookSize = 150 # the size of codebook

is_cv = False

svm_c = 512.0
svm_g = 0.0001220703125
cv_acc = 79.2956