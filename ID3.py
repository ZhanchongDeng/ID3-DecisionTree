'''
Contains all methods for a ID3 Decision Tree.
@Author: Zhanchong Deng
@Date: 2/15/2020
'''
import numpy as np
import pandas as pd

'''
Return numpy arrays representing the dataset.
@param fp is the file path.
@return 2D numpy array, (#entry, 23).
'''
def loadData(fp):
    train_file = open(fp, 'r')
    train_file.seek(0)
    raw_strings = train_file.read().split("\n")[:-1]
    return np.array([np.array(entry[0:-1].split(" "), dtype="float") for entry in raw_strings])


'''
Decision Tree Object.
'''
class id3:
    # Default Constructor
    def __init__(self):
        print("Created a new ID3 tree")
    
    # Training
    # To String
    # Pruning
    # Train
    def fit(self, training_data, labels):
        self.root = idNode(None, None, -1, -1)
        self.root.isLeaf = False
        self.root = self.build(self.root, training_data)
    
    def build(self, node, v):
        # Stop if it is pure
        if self.isPure(v):
            # Make it a leaf node.
            newnode.isLeaf = True
            newnode.rule = v[0][-1]
        # Parse them according to H(entropy)
        else:
            # Pick a feature f and threshold t
            feature = 0
            threshold = self.generateThreshold(v[:,[feature,-1]])
            # Split them according to the rule
            v_yes = v[v[:,-1] >= threshold]
            v_no = v[v[:,-1] < threshold]
            # Create new nodes
            node.yes = idNode(None, None, -1, -1)
            self.build(node.yes, v_yes)
            node.no = idNode(None, None, -1, -1)
            self.build(node.no, v_no)
            
      
    def isPure(self, vec):
        return len(np.unique(vec[:,-1])) == 1
            
    def generateThreshold(self, feature):
        # Sort them according to given feature
        sortedFeatures = feature[feature[:,0].argsort()]
        # Determine where the labels switched
        changed = sortedFeatures[:-1,-1] != sortedFeatures[1:,-1]
        # Get the before and after value at those swithces
        before = sortedFeatures[:-1,0]
        after = sortedFeatures[1:,0]
        # Return a list of the threshold by this feature
        return np.unique((before[changed] + after[changed]) / 2)

    
'''
Node for Decision Tree
'''
class idNode:
    # Fields: yes(idNode), no(idNode), data(ndarray), labels(ndarray), threshold(float), feature(int)
    def __init__(self, yes, no,threshold, feature):
        self.yes = yes
        self.no = no
        self.threshold = threshold
        self.feature = feature