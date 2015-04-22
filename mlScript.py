# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 12:52:17 2015

@author: pnguye41
"""
import pickle
import ml
import numpy as np

from sklearn import cross_validation

##This is now pickled

#aa = np.loadtxt(open("outputFinalNoRow.csv","rb"),delimiter=",")
#aa[aa==-1] = numpy.nan
#allMatrix = aa[:,1:]
#
##The response Variables are in the middle, MACCE, death,
##split the matrix into three to extract the middle responses
#
#    
#complications = allMatrix[:,24:38]
#leftMatrix = allMatrix[:,0:24]
#rightMatrix = allMatrix[:,38:]
#featuresMatrix =np.concatenate((leftMatrix,rightMatrix),axis=1) 
#deathVector = complications[:,13]
#anyComp = (np.any(complications,axis = 1)).astype(float)
#
#with open("mlData.dat","wb") as f:
#    pickle.dump(allMatrix, f)
#    pickle.dump(complications, f)
#    pickle.dump(featuresMatrix, f)
#    pickle.dump(deathVector, f)
#    pickle.dump(anyComp, f)
    

#Load Data
with open("mlData.dat","r") as f: #try "rb" or "r", depending on computer
    allMatrix = pickle.load(f)
    complications =pickle.load(f)
    featuresMatrix =pickle.load(f)
    deathVector = pickle.load(f)
    anyComp = pickle.load(f)
    
#remove first hypertension vector
#featuresMatrix = featuresMatrix[:,1:]
    
#use only ER variables
featuresMatrix = featuresMatrix[:,[1,4,24,26,28,29]]
    
##choose one below to be the response vector
#response = anyComp
response = deathVector

#Choose number of folds
nFolds = 5;

###Choose Type of Classified
#type of classifier, choose, modify, comment out/in, one of the classifiers below
#SVC classifier
#
#classifierType = 'svc'
#cRange = np.linspace(1,4,10)
#gammaRange = np.logspace(-4,-2,10)
##cRange = [6]
##gammaRange = [0.001]
#
#hList = [[cRange,gammaRange],['C','gamma']]

##SVC classifier with varying SMOTE
#
#classifierType = 'svc'
#cRange = np.linspace(0.5,5,10)
#gammaRange = np.logspace(-4,-2,10)
##cRange = [2.66666666667]
##gammaRange = [0.00046415888336127773]
#minorityPercent = [100]
#majorityPercent = [0]
##minorityPercent = np.linspace(50,300,6)
##majorityPercent = np.linspace(0,90,10)
#hList = [[cRange,gammaRange,minorityPercent,majorityPercent],['C','gamma','minorityPercent','majorityPercent']]


#SVC Linear
##
classifierType = 'svcLinear'
#cRange = np.linspace(4,30,10)
cRange = np.logspace(-5,5,50)
#penaltyType = ['l1','l2']
#cRange = [0.0003]
penaltyType = ['l1']
hList = [[cRange,penaltyType],['C','penalty']]

#adaboost SVC
#
#classifierType = 'svcAdaboost'
##cRange = np.linspace(4,30,10)
##gammaRange = np.logspace(-3,-1,10)
#cRange = [4]
#gammaRange = [0.01]
#hList = [[cRange,gammaRange],['C','gamma']]

#adaboost decision Tree
#
#classifierType = 'dTreeAdaboost'

#neural net




###end of classifier choice



#create test set, use 1/5th of the data as test set
#using the method similar to cv to get test set
#true cross validation will still be used in the runClassifier
skf = cross_validation.StratifiedKFold(response, n_folds=5,shuffle= True,random_state=0)
trainIndices,testIndices = iter(skf).next()
trainFeatures = featuresMatrix[trainIndices]
trainResponses = response[trainIndices]
testFeatures = featuresMatrix[testIndices]
testResponses= response[testIndices]



################################################
#Run the Classifier
accAll, fAll, rocAll,hParams = ml.runCVClassifier(trainFeatures,trainResponses,nFolds,classifierType,hList)

#Choose the hyperparameters that maximize the metric
bestIndex = np.argmax(rocAll)
bestParams = list(hParams[bestIndex])
#Re-train the classifier on the training set using the chosen hyper parameters, and test on the test set
accTest,fTest,rocTest,clf = ml.runTestClassifier(trainFeatures,testFeatures,trainResponses,testResponses,classifierType,[bestParams,hList[1]])

