#*******************************************************************
# Sistemas Distribuidos
#
# Description: Implements the k-nearest neighbors classifier and tests it
#*******************************************************************

import time
import numpy as np
import statistics as stat
import scipy.spatial.distance as dist

def n_validator(data, p, classifier, *args):
    '''The purpose of this function is to estimate the performance of a
    classifier in a particular setting. This function takes as input an m x 
    (n+1) array of data, an integer p, the classifier it is checking, and any 
    remaining parameters that the classifier requires will be stored in args. 
    Returns the estimate of the classifier's performance on data from this 
    source.'''
    
    # Shuffling and splitting array in p sections
    np.random.shuffle(data)
    m = data.shape[0]
    n = data.shape[1] - 1
    sections = np.array_split(data, p)
        
    success = 0
    # Testing for each section
    for i in range(p):
        aux = sections[:]
        aux.pop(i)
    
        # Constructing function arguments
        f_args = (np.vstack(aux), np.delete(sections[i], n, 1)) + args
#        print("Training: ")    
#        print(np.vstack(aux))
#        print("Test: ")
#        print(np.delete(sections[i], n, 1))
        
        labels = classifier (*f_args)
        
#        print("Labels: ")
#        print(labels)
#        
#        print("Sections: ")
#        print(sections[i])
                
        # Computing success
        for k in range(labels.size):
            if labels[k] == sections[i][k][n]:
                success += 1
    
    return success / m


def KNNclassifier(training, test, k, d, *args):
    '''Implements the k-nearest neighbors classifier, using a training data set
    and the test data set to be labeled. Receives k by argument, as well as the
    distance function to be used. Any other arguments that might be needed by
    the distance function are stored in *args'''
    
    # Saving dimensions
    q = training.shape[0]
    n = training.shape[1] - 1
    j = test.shape[0]
    
    # Removing labels to compute distance
    aux = np.delete(training, n, 1)
        
    labels = []
    # Finding the k-nearest neighbors for each array in test
    for i in range(j):
        # List with k-nearest neighbors
        k_labels = []
        # Array to store the distances
        neighbors_dist = np.array([])
        # Array to store the indices
        neighbors_index = np.array([])
        # Keep track of more distant neighbor
        max_neighbor = 0
        
        # Insert first k elements
        for count in range(k):
            # Constructing function arguments
            d_args = (test[i], aux[count]) + args
                        
            neighbors_dist = \
                np.insert(neighbors_dist, 0, d(*d_args))
            neighbors_index = np.insert(neighbors_index, 0, count)
        max_neighbor = max(neighbors_dist)
        
        # Run through the rest
        for count in range(k, q):
            # Constructing function arguments
            d_args = (test[i], aux[count]) + args
            
            new_dist = d(*d_args)
            # Then only insert if it is smaller than one of the k selected
            if (new_dist < max_neighbor):
                # Get the index of ocurrence of max element
                max_index = np.where(neighbors_dist == max_neighbor)[0][0]
                # Remove old max
                neighbors_dist = np.delete(neighbors_dist, max_index)
                neighbors_index = np.delete(neighbors_index, max_index)
                # Insert new one
                neighbors_dist = np.insert(neighbors_dist, 0, new_dist)
                neighbors_index = np.insert(neighbors_index, 0, count)
                # Update max
                max_neighbor = max(neighbors_dist)
        
        # Use indices to find most common label
        for index in neighbors_index:
            k_labels.append(training[index][n])
        
        # Use the mode as label            
        labels.append(stat.mode(k_labels))
        
    return np.array(labels)
    

def real_data():
    '''Read from a file to create and return an array of data to test the 
    knn.py module'''
    
    f = open ("adult-reduced.data")
    
    count = 0
    data = []
    # Read data from file
    for line in f:
        count += 1
        # Remove \n, split using "," as separator and remove first field (ID)
        observation = line.strip().split(',')[:]
        print(observation[-1])
        # Append label to the end
        if (observation[-1] == " >50K"):
            observation.append('1')
        else:
            observation.append('0')
                  
        # Including only relevant continuous data 
        t = observation[0:1] + observation[4:5] + observation[10:13] + \
                    observation[-1:]
        
        # Remove unknown tuples
        if (t.count('?') == 0 and len(t) == 6):
            data.append(t[:])
                
    f.close()
    
    for i in data:
        print(i[-1])
        
    return np.array(data, dtype=float)

def main():
    '''Main function used to test the KNNclassifier using two different data
    sets, one made out of real data, another one generated using multivariate
    normal distribution.'''
    
    print(time.strftime("%H:%M:%S"))
    
    # Distance being used to test
    dist_func = dist.minkowski    
    
    # Parameter for minkowski distance
    p = 3
    
    # Testing for real data
    data = real_data()
    
    success = []
    # Testing the classifier for different values of k
    for k in range(1,16,2):
        success.append\
            (n_validator(data, 5, KNNclassifier, k, dist_func, p))
    
    # Finding best k for real data
    best_k = success.index(max(success)) * 2 + 1
    
    print("Best k for real data:", best_k, "(", '{:.2}'.format(max(success)),\
            ")")
    
    print(time.strftime("%H:%M:%S"))
        

main()