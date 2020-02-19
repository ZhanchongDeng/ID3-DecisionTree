'''
Contains all methods for a ID3 Decision Tree.
@Author: Zhanchong Deng
@Date: 2/15/2020
'''
import numpy as np
import pandas as pd
from scipy.stats import entropy

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
Node for Decision Tree
'''


class idNode():
    # Fields: yes(idNode), no(idNode), threshold(float), feature(int)
    # Branches
    yes = None
    no = None
    # Is is a leaf?
    isLeaf = False
    # How to get here
    route = "root"
    # For non Leaves
    threshold = -1
    feature = -1
    # For leaves
    rule = -1
    
    def toString(self):
        if not self.isLeaf:
            return "(" + self.route + ")" + "[is feature at " + str(self.feature+1) + " >= " + str(self.threshold) \
                   + "?]\n"
        else:
            return "(" + self.route + ")" + "[" + str(self.rule) + "]\n"

        
'''
Decision Tree Object.
'''


class id3():
    # Default Constructor
    def __init__(self):
        self.root = idNode()
        print("Created a new ID3 tree")

    def fit(self, training_data):
        self.build(self.root, training_data)

    def build(self, node, v):
        # Stop if it is pure
        if self.isPure(v):
            # Make it a leaf node.
            node.isLeaf = True
            node.rule = v[0][-1]
            
        # Parse them according to H(entropy)
        else:
            # Pick a feature f and threshold t
            node.feature = 0
            minEntropy = float("inf")
            for thisfeature in range(0,len(v[0]) - 1):
                v_at_f = v[:, [thisfeature, -1]]
                list_of_threshold = self.generateThreshold(v_at_f)
                # This feature is not fit for splitting, pick another
                if len(list_of_threshold) == 0:
                    continue
                # Set Threshold as the current best
                result = self.maxIG(list_of_threshold, v_at_f)
                if result[1] < minEntropy:
                    minEntropy = result[1]
                    node.feature = thisfeature
                    node.threshold = result[0]
            # Split them according to the rule
            v_yes = v[v[:,node.feature] >= node.threshold]
            v_no = v[v[:,node.feature] < node.threshold]
            # Create new nodes
            node.yes = idNode()
            node.yes.route = "yes"
            self.build(node.yes, v_yes)
            node.no = idNode()
            node.no.route = "no"
            self.build(node.no, v_no)

    def isPure(self, vec):
        return len(np.unique(vec[:, -1])) == 1

    def generateThreshold(self, feature):
        distinct_val = np.sort(np.unique(feature[:,0]))
        if len(distinct_val) == 0:
            print(feature)
        return (distinct_val[:-1] + distinct_val[1:]) / 2
    # def generateThreshold(self, feature):
    #     # Sort them according to given feature
    #     sortedFeatures = feature[feature[:, 0].argsort()]
    #     # Determine where the labels switched
    #     changed = sortedFeatures[:-1, -1] != sortedFeatures[1:, -1]
    #     # Get the before and after value at those swithces
    #     before = sortedFeatures[:-1, 0]
    #     after = sortedFeatures[1:, 0]
    #     # Return a list of the threshold by this feature
    #     return np.unique((before[changed] + after[changed]) / 2)

    
    def maxIG(self, list_of_threshold, v_at_f):
        # Initialize as the first threshold
        minthreshold = list_of_threshold[0]
        minEntropy = self.entropy_with_threshold(minthreshold, v_at_f)
        for i in range(1,len(list_of_threshold)):
            newEntropy = self.entropy_with_threshold(list_of_threshold[i], v_at_f)
            # update the threshold with the minimum entropy
            if minEntropy > newEntropy:
                minthreshold = list_of_threshold[i]
                minEntropy = newEntropy
        return (minthreshold, minEntropy)


    def entropy_with_threshold(self, threshold, v_at_f):
        # Slice v according to the given threshold
        v_yes = v_at_f[v_at_f[:,0] >= threshold]      
        v_no = v_at_f[v_at_f[:,0] < threshold]
        # Calculate H(X|Z=yes)
        margin_yes = [ np.sum(v_yes[:,-1]==num_labels) / len(v_yes) for num_labels in np.unique(v_yes[:,-1])]
        h_yes = entropy(margin_yes)
        # Calculate H(X|Z=no)
        margin_no = [ np.sum(v_no[:,-1]==num_labels) / len(v_no) for num_labels in np.unique(v_no[:,-1])]
        h_no = entropy(margin_no)
        # Calculate H(X|Z)
        return len(v_yes)/len(v_at_f) * h_yes + len(v_no)/len(v_at_f) * h_no
    
    def toString(self):
        tree_str = self.printTree(self.root, 0)
        return tree_str
    
    def printTree(self, curNode, level):
        # do yourself
        curStr = '\t' * level + curNode.toString()
        # do yes
        if not curNode.yes.isLeaf:
            curStr += self.printTree(curNode.yes, level + 1)
        else:
            curStr += '\t' * (level + 1) + curNode.yes.toString()
        # do no
        if not curNode.no.isLeaf:
            curStr += self.printTree(curNode.no, level + 1)
        else:
            curStr += '\t' * (level + 1) + curNode.no.toString()
        return curStr