# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 03:45:23 2015

@author: pnguye41
"""

import numpy as np
import smote
import itertools


from sklearn import preprocessing as prep
from sklearn.pipeline import make_pipeline
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn import ensemble



def cartesian(arrays, out=None):
    #from http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def fitMACCEsvm(X_train,y_train,classifierType = 'svc', params = []):

    #split data by class,
    haveResponse = X_train[y_train==1]
    noResponse = X_train[y_train==0]
    #imputer (fill in missing data)
    haveImp = prep.Imputer(missing_values="NaN", strategy='mean', axis=0)
    noImp = prep.Imputer(missing_values="NaN", strategy='mean', axis=0)
    
    haveImp.fit(haveResponse)
    noImp.fit(noResponse)
    
    haveR = haveImp.transform(haveResponse)
    noR = noImp.transform(noResponse)

    # SMOTE, and undersampling, non-hyperparameter
    minorityPercent = 100
    imbalance = int((minorityPercent/100.0)*round(len(noR)/(len(haveR))))*100
    newSamples = smote.SMOTE(haveR,imbalance,5)
    fsHR = np.concatenate((haveR,newSamples))
    majorityPercent = 0
    removeIndices = np.random.choice(len(noR),int((majorityPercent/100.0)*len(noR)),replace=False)
    noR = np.delete(noR,removeIndices,axis = 0)
    
    #SMOTE ONLY
#    minorityPercent = 100;
#    imbalance = int((minorityPercent/100.0)*round(len(noR)/(len(haveR))))*100
#    newSamples = smote.SMOTE(haveR,imbalance,5)
#    fsHR = np.concatenate((haveR,newSamples))
 

    #NOSMOTE case
#    fsHR = haveR
    
    #SMOTE AND hyperparameter UNDERSAMPLING
#    minorityPercent =params[2];
#    imbalance = int((minorityPercent/100.0)*round(len(noR)/(len(haveR))))*100
#    newSamples = smote.SMOTE(haveR,imbalance,5)
#    fsHR = np.concatenate((haveR,newSamples))
#    majorityPercent = params[3]
#    removeIndices = np.random.choice(len(noR),int((majorityPercent/100.0)*len(noR)),replace=False)
#    noR = np.delete(noR,removeIndices,axis = 0)
    
    #JUST UNDERSAMPLING
#    fsHR = haveR
#    majorityPercent = (len(noR)/float(len(noR)+len(haveR)))*100.0
#    removeIndices = np.random.choice(len(noR),int((majorityPercent/100.0)*len(noR)),replace=False)
#    noR = np.delete(noR,removeIndices,axis = 0)

    #recombine the two matrices
    currFeats = np.concatenate((fsHR,noR))
    currResps = np.concatenate((np.ones(len(fsHR)),np.zeros(len(noR))))
    
    #fill in 
    imputer = prep.Imputer(missing_values = "NaN",strategy="mean",axis=0)
    imputer.fit(currFeats)
    
    if classifierType=='svc':
        classifier = make_pipeline(prep.StandardScaler(), svm.SVC(kernel='poly',degree=4,gamma=params[1],C=params[0]))
        
    elif classifierType=='svcLinear':
        classifier = make_pipeline(prep.StandardScaler(), svm.LinearSVC(C=params[0],penalty = params[1],dual=False))
    elif classifierType =='svcAdaboost':
        baseSVM = svm.SVC(kernel='rbf',gamma=params[1],C=params[0])
        adaSVM = ensemble.AdaBoostClassifier(base_estimator = baseSVM,algorithm = 'SAMME')
        classifier = make_pipeline(prep.StandardScaler(),adaSVM)
    elif classifierType =='svcLinearAdaboost':
        baseSVM = svm.LinearSVC(C=params[0],penalty = params[1],dual=False)
        adaSVM = ensemble.AdaBoostClassifier(base_estimator = baseSVM,algorithm = 'SAMME')
        classifier = make_pipeline(prep.StandardScaler(),adaSVM)
        
    classifier.fit(currFeats,currResps)
    return classifier,imputer
    




def runCVClassifier(featuresMatrix,response,nFolds,classifierType,hList):
    
    #Generate Hyper parameter list
    
    hyperParams = []

    for i in itertools.product(*hList[0]):
        hyperParams.append(i)

        
    #Set up metric arrays
    accAll = np.zeros(len(hyperParams))
    fAll = np.zeros(len(hyperParams))
    rocAll = np.zeros(len(hyperParams))
    
    #run the classification
    for j in range(len(hyperParams)):
        print 'currently trying:'
        for paramNum in range(len(hList[0])):
            print  ' ' + str(hList[1][paramNum]) + ' = ' + str(hyperParams[j][paramNum]),
        print
        skf = cross_validation.StratifiedKFold(response, n_folds=nFolds,shuffle= True)
        
        accArray = []
        fArray = []
        rocArray = []
        #for each k-fold
        for train_index, test_index in skf:
            X_train, X_test = featuresMatrix[train_index], featuresMatrix[test_index]
            y_train, y_test = response[train_index], response[test_index]
        
            
            clf,fullImp = fitMACCEsvm(X_train,y_train,classifierType,hyperParams[j])
            X_test = fullImp.transform(X_test)
            y_hat = clf.predict(X_test)
            
            
            '''Metrics on prediction
            '''
            correct = (np.equal(y_hat,y_test)).astype('float')
            acc = sum(correct)/len(correct)
            fScore = metrics.f1_score(y_test,y_hat)
            roc_score = metrics.roc_auc_score(y_test,y_hat)
            
            accArray.append(acc)
            fArray.append(fScore)
            rocArray.append(roc_score)
        
        accAll[j] = np.mean(accArray)
        fAll[j] = np.mean(fArray)
        rocAll[j] = np.mean(rocArray)
        
        print 'accuracy = '+  str(accAll[j]) +  '  F = '+ str(fAll[j])+ '  ROC = '+ str(rocAll[j])
    
    
    
    return accAll, fAll, rocAll,hyperParams
    
def runTestClassifier(X_train,X_test,y_train,y_test,classifierType,hList):
    
    #Generate Hyper parameter list
    

    hyperParams = [hList[0]]
    #run the classification
    
    print 'Testing Set'
    
    #for each k-fold

    
    
    clf,fullImp = fitMACCEsvm(X_train,y_train,classifierType,hyperParams[0])
    X_test = fullImp.transform(X_test)
    y_hat = clf.predict(X_test)
    
    
    '''Metrics on prediction
    '''
    correct = (np.equal(y_hat,y_test)).astype('float')
    acc = sum(correct)/len(correct)
    fScore = metrics.f1_score(y_test,y_hat)
    roc_score = metrics.roc_auc_score(y_test,y_hat)
    
    print 'accuracy = '+  str(acc) +  '  F = '+ str(fScore)+ '  ROC = '+ str(roc_score)
    
    
    
    return acc,fScore,roc_score,clf
    
