#*******************************************************************
# Sistemas Distribuidos 
#
# Description: Implements the k-nearest neighbors classifier with Spark 
# and tests it
#*******************************************************************

import time
import math
import numpy as np
import statistics as stat
import scipy.spatial.distance as dist
from random import shuffle
from pyspark import SparkConf, SparkContext

def euclidean_distance (a, b):
    '''Calculates the euclidean distance between two points in a n-dimensional
    space. Takes two lists with the points' cordinates as parameters and returns
    the distance between them as a float.'''
    c = []
    aux = 0
    for i in range(0, len(a)):
        aux += ((a[i] - b[i]) ** 2)
        
    return math.sqrt(aux)

def n_validator(data, p, classifier, *args):
    '''The purpose of this function is to estimate the performance of a
    classifier in a particular setting. This function takes as input an m x 
    (n+1) array of data, an integer p, the classifier it is checking, and any 
    remaining parameters that the classifier requires will be stored in args. 
    Returns the estimate of the classifier's performance on data from this 
    source.'''
    
    # Shuffling and splitting list in p sections
    shuffle(data)
    m = len(data)
    n = len(data[0]) - 1
            
    unlabeled = []
    labels = []
    # Create unlabeled list for test set and labels
    for i in range(0, m):
        unlabeled.append(data[i][0:-1])
        labels.append(data[i][-1])
    
    lab_sections = []
    unlab_sections = []
    # Partioning the data into p different sections
    for i in range(0, m, int(m/p)):
        lab_sections.append(labels[i:int(i+m/p)])
        unlab_sections.append(unlabeled[i:int(i+m/p)])
        
        
    success = 0
    # Testing for each section
    for i in range(p):
        aux = []
        auxLab = []
        # Removing unlab_sections[i] from training set and joining other sections
        for j in range(p):
            if (i != j):
                for k in unlab_sections[j]:
                    aux.append(k)
                for k in lab_sections[j]:
                    auxLab.append(k)
                    
        # Constructing function arguments
        f_args = (aux, unlab_sections[i], auxLab) + args      
        
        labelsC = classifier (*f_args)
                
        # Computing success
        for k in range(len(labelsC)):
            if labelsC[k] == lab_sections[i][k]:
                success += 1
    
    return success / m

def KNNclassifier(training, test, tLabels, k, d, *args):
    '''Implements the k-nearest neighbors classifier, using a training data set
    and the test data set to be labeled. Receives k by argument, as well as the
    distance function to be used. Any other arguments that might be needed by
    the distance function are stored in *args'''
    
    # Saving dimensions
    q = len(training)
    n = len(training[0]) - 1
    j = len(test)
    
    
    trainingRDD = sc.parallelize(training)
    
    labels = []
    for i in test:
        dist = trainingRDD.map(lambda x: euclidean_distance(x, i)).collect()
        
        k_labels = []
        # Getting labels of k-nearest neighbors
        for i in range(k):
            nNeighbor = min(dist)
            nnIndex = dist.index(nNeighbor)
            k_labels.append(tLabels[nnIndex])
            dist.remove(nNeighbor)
        
        labels.append(stat.mode(k_labels))
    
    return labels
    

def real_data():
    '''Read from a file to create and return an array of data to test the 
    knn.py module'''
    
    f = open ("adult-reduced.data")
    
    data = []
    # Read data from file
    for line in f:

        # Remove \n, split using "," as separator and remove first field (ID)
        observation = line.strip().split(',')[:]
        
        # Append label to the end
        if (observation[-1] == ' >50K'):
            observation.append('1')
        else:
            observation.append('0')
                  
        # Including only relevant continuous data 
        t = observation[0:1] + observation[4:5] + observation[10:13] + \
                    observation[-1:]
        
        # Remove unknown tuples
        if (t.count('?') == 0 and len(t) == 6):
            data.append([ float(i) for i in t[:]])
                
    f.close()
    
    return data

def main():
    '''Main function used to test the KNNclassifier using two different data
    sets, one made out of real data, another one generated using multivariate
    normal distribution.'''
    
    print(time.strftime("%H:%M:%S"))
    
    # Distance being used to test
    dist_func = euclidean_distance  
    
    # Testing for real data
    data = real_data()
    
    success = []
    # Testing the classifier for different values of k
    for k in range(1,16,2):
        success.append\
            (n_validator(data, 5, KNNclassifier, k, dist_func))
    
    # Finding best k for real data
    best_k = success.index(max(success)) * 2 + 1

    out = open ("out", "w")
    
    out.write("Best k for real data: {} ({:.2})\n".format(best_k, max(success)))

    
    print(time.strftime("%H:%M:%S"))
        
conf = SparkConf().setAppName("K-Nearest Neighbors").setMaster("spark://frontend:7077")
sc = SparkContext(conf=conf)
main()