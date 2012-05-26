'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington

Things to note:
    Entropy is a measurement of "chaos in a set"
    
    Example high entropy set:
    high_entropy_set = set("dog", "cat", "bird", "fish", "lizard")
    
    Example low entropy set:
    low_entropy_set = set("dog", "cat", "dog", "dog", "cat")
    
    Example 0 entropy set:
    zero_entropy_set = set("dog", "dog", "dog", "dog", "dog")
    
    Each set has the same number of values, but the low entropy set has less 
    "chaos". The last set is uniform (all values are the same) so it has an
    entropy of 0.
    
'''

from math import log
import operator
from collections import Counter

def createDataSet():
    labels = [      'no surfacing', 'flippers'] # is fish
    dataSet = [
               [    1,              1,          'yes'   ],
               [    1,              1,          'yes'   ],
               [    1,              0,          'no'    ],
               [    0,              1,          'no'    ],
               [    0,              1,          'no'    ],
    ]
    
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = Counter(featVec[-1] for featVec in dataSet)
    
    logBase2 = lambda n: log(n, 2)
    def entropy(labelCount):
        prob = float(labelCount) / numEntries
        return -1 * prob * logBase2(prob)
    
    return sum(entropy(labelCount) for labelCount in labelCounts.values())
    
def splitDataSet(dataSet, axis, value):
    return [ 
        # chop out axis used for splitting:
        featVec[:axis] + featVec[axis+1:]
        
        for featVec in dataSet
        if featVec[axis] == value ]
    
def chooseBestFeatureToSplit(dataSet):
    # The last column is used for the labels
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    
    def infoGain(featureToSplitOn):
        # get a set of unique values for this feature
        uniqueVals = set(example[i] for example in dataSet)    
        
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            
        # calculate the info gain; ie reduction in entropy
        return baseEntropy - newEntropy
    
    features = range(numFeatures)
    return max(features, key=infoGain)

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
