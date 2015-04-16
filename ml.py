# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 03:45:23 2015

@author: pnguye41
"""
import sklearn
import numpy as np
import smote
import pickle
import time
import itertools


from sklearn import preprocessing as prep
from sklearn.pipeline import make_pipeline
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics


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

    #SMOTE
    imbalance = int(round(100*len(noR)/(100*len(haveR)))*100)
    newSamples = smote.SMOTE(haveR,imbalance,5)
    fsHR = np.concatenate((haveR,newSamples))

    #recombine the two matrices
    currFeats = np.concatenate((fsHR,noR))
    currResps = np.concatenate((np.ones(len(fsHR)),np.zeros(len(noR))))
    
    #fill in 
    imputer = prep.Imputer(missing_values = "NaN",strategy="mean",axis=0)
    imputer.fit(currFeats)
    
    if classifierType=='svc':
        classifier = make_pipeline(prep.StandardScaler(), svm.SVC(kernel='rbf',gamma=params[1],C=params[0]))
        
    elif classifierType=='svcLinear':
        classifier = make_pipeline(prep.StandardScaler(), svm.LinearSVC(C=params[0],penalty = params[1],dual=False))
        
    classifier.fit(currFeats,currResps)
    return classifier,imputer
    


'''Start of Main function

'''

#Load Data
with open("mlData.dat","rb") as f:
    allMatrix = pickle.load(f)
    complications =pickle.load(f)
    featuresMatrix =pickle.load(f)
    deathVector = pickle.load(f)
    anyComp = pickle.load(f)
    
##choose one below to be the response vector
response = anyComp
#response = deathVector

#Choose number of folds
nFolds = 5;

###Choose Type of Classified
#type of classifier, choose, modify, comment out/in, one of the classifiers below
#SVC classifier

#classifierType = 'svc'
#cRange = np.linspace(4,30,10)
#gammaRange = np.logspace(-3,-1,10)
##cRange = [10000]
##gammaRange = [0.0001]
#hList = [cRange,gammaRange]


#SVC Linear

classifierType = 'svcLinear'
#cRange = np.linspace(4,30,10)
#penaltyType = ['l1','l2']
cRange = [5]
penaltyType = ['l1']
hList = [cRange,penaltyType]



###end of classifier choice

#Generate Hyper parameter list
hyperParams = []
for i in itertools.product(*hList):
    hyperParams.append(i)
    
#Set up metric arrays
accAll = np.zeros(len(hyperParams))
fAll = np.zeros(len(hyperParams))
rocAll = np.zeros(len(hyperParams))

#run the classification
for j in range(len(hyperParams)):
    print 'currently trying  C = ' + str(hyperParams[j][0]) +',param2 =  ' + str(hyperParams[j][1]) + ',',
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






