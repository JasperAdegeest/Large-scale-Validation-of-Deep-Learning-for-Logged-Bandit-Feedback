import collections
import gzip
import Instance
import math
import numpy
import os
import scipy.sparse
import scipy.misc
import sklearn.preprocessing
import sys


class Dataset:
    def __init__(self):
        self.instanceList = None
        
    def create_synthetic_data(self, num_records, num_features, num_labels):
        if num_records <= 0 or num_features <= 0 or num_labels <= 0:
            print("Dataset:create_synthetic_data\t[ERR]\tCannot create synthetic data with numRecords/numFeatures/numLabels ",
                    num_records, num_features, num_labels, flush=True)
            sys.exit(0)

        features = numpy.random.randn(num_records, num_features)
        features = sklearn.preprocessing.robust_scale(features)
        features = sklearn.preprocessing.normalize(features)
        
        labels = numpy.random.randint(0, high = 2, size = (num_records, num_labels))
        print("Dataset:create_synthetic_data\t[LOG]\tCreated synthetic data with numRecords/numFeatures/numLabels ", num_records, num_features, num_labels, flush=True)
        return features, labels
    
    def write_bandit_data(self, repo_dir, features, labels, instance_type, seed):
        fileName = repo_dir + instance_type + '.txt'
        if not os.path.exists(repo_dir):
            print("Dataset:write_bandit_data\t[ERR]\tOutput directory not found at ", repo_dir, flush=True)
            sys.exit(0)
        
        numpy.random.seed(seed)
        
        numRecords, numFeatures = numpy.shape(features)
        numLabels = numpy.shape(labels)[1]
        
        f = open(fileName, 'w')
        #Write header
        if instance_type == 'MultiClass' or instance_type == 'MultiLabel':
            f.write(str(numRecords) + ' ' + str(numFeatures) + ' ' + str(numLabels) + '\t# HEADER: numRecords numFeatures numLabels\n')
        elif instance_type == 'Brute':
            f.write(str(numRecords) + ' ' + str(numLabels * numFeatures) + ' ' + str(numLabels) + \
                    '\t# HEADER: numRecords numFeatures maxPossibleActions\n')
        elif instance_type == 'PlackettLuce':
            f.write(str(numRecords) + ' ' + str(numFeatures) + ' ' + str(numLabels) + ' ' + str(numLabels) + \
                    '\t# HEADER: numRecords numFeatures maxCandidates maxSlots\n')
        else:
            print("Dataset:write_bandit_data\t[ERR]\tinstance_type not recognized: ", instance_type, flush=True)
            sys.exit(0)
        
        for i in range(numRecords):
            featureStr = ''
            for j in range(numFeatures):
                if features[i,j] != 0:
                    featureStr += str(j) + ':' + str(features[i,j]) + ' '
                    
            featureStr += '\t# ID:' + str(i) + '\n'
                    
            instanceStr = ''
            if instance_type == 'MultiClass':
                sampledClass = numpy.random.randint(numLabels)
                sampledLoss = - labels[i, sampledClass]
                propensity = 1.0 / numLabels
                instanceStr = str(sampledClass) + ' ' + str(sampledLoss) + ' ' + str(propensity) + ' '
                
            elif instance_type == 'MultiLabel':
                sampledLabels = numpy.random.randint(0, high = 2, size = numLabels)
                sampledLoss = (labels[i, :] != sampledLabels).sum(dtype = numpy.longdouble) - numLabels
                propensity = 1.0 / (2 ** numLabels)
                labelStr = ''
                for k in range(numLabels):
                    if sampledLabels[k] > 0:
                        labelStr += str(k) + ','
                        
                if labelStr == '':
                    labelStr = '-1'
                else:
                    labelStr = labelStr[:-1]    
                instanceStr = labelStr + ' ' + str(sampledLoss) + ' ' + str(propensity) + ' '
                
            elif instance_type == 'Brute':
                numAllowedActions = numpy.random.randint(2, high = numLabels)
                allowedActions = numpy.random.choice(numLabels, size = numAllowedActions, replace = False)

                sampledClassIndex = numpy.random.randint(numAllowedActions)
                sampledClass = allowedActions[sampledClassIndex]
                sampledLoss = - labels[i, sampledClass]
                propensity = 1.0 / numAllowedActions
                instanceStr = str(sampledClassIndex) + ' ' + str(sampledLoss) + ' ' + str(propensity) + ' '
                
                featureStr = str(numAllowedActions) + '\t# ID:' + str(i) + ' -- ' + numpy.array_str(allowedActions) + '\n'
                for k in range(numAllowedActions):
                    featureStr += str(k) + ' '
                    currentAction = allowedActions[k]
                    for j in range(numFeatures):
                        if features[i,j] != 0:
                            featureStr += str(currentAction * numFeatures + j) + ':' + str(features[i,j]) + ' '
                    featureStr += '\t# Action: ' + str(currentAction) + '\n'
            
            elif instance_type == 'PlackettLuce':
                numSlots = numpy.random.randint(1, high = numLabels)
                sampledSlate = numpy.random.permutation(numLabels)[0:numSlots]
                sampledLabels = labels[i, sampledSlate]
                positionWeights = numpy.array(range(numSlots, 0, -1))
                
                weightedGain = numpy.dot(positionWeights, sampledLabels)
                normalizer = numSlots * (numSlots + 1) / 2
                normalizedGain = (1.0 * weightedGain) / normalizer
                
                sampledLoss = - normalizedGain
                
                propensity = 1.0 / (scipy.misc.comb(numLabels, numSlots) * math.factorial(numSlots))
                labelStr = ''
                for k in sampledSlate:
                    labelStr += str(k) + ','
                
                instanceStr = labelStr[:-1] + ' ' + str(sampledLoss) + ' ' + str(propensity) + ' '
            
                featureStr = str(numLabels) + '\t# ID:' + str(i) + '\n'
                for k in range(numLabels):
                    featureStr += str(k) + ' '
                    for j in range(numFeatures):
                        if features[i,j] != 0:
                            featureStr += str(k*numFeatures + j) + ':' + str(features[i,j]) + ' '
                    featureStr += '\t# Action: ' + str(k) + '\n'
            
            f.write(instanceStr + featureStr)
                
        f.close()
        print("Dataset:write_bandit_data\t[LOG]\tOutput bandit data to ", fileName, flush=True)
        
    def read_bandit_data(self, repo_dir, instance_type):
        if repo_dir.endswith('.txt'):
            fileName = repo_dir
        else:
            fileName = repo_dir + instance_type+'.txt'
            
        if not os.path.exists(fileName):
            print("Dataset:read_bandit_data\t[ERR]\tInput file not found at ", fileName, flush=True)
            sys.exit(0)
    
        f = open(fileName, 'r')
        allLines = f.readlines()
        f.close()
        
        header = allLines[0]
        commentIndex = header.find('#')
        if commentIndex >= 0:
            header = header[:commentIndex]
        tokens = header.split()
        numRecords = int(tokens[0])
        numFeatures = int(tokens[1])
        numLabels = int(tokens[2])
        
        currIndex = 0
        instanceList = []
        print("Dataset:read_bandit_data\t[LOG]\tFilename: %s Number of instances: %d" %\
                (fileName, numRecords), flush=True)
                
        for i in range(numRecords):
            currIndex += 1
            currentLine = allLines[currIndex]
            commentIndex = currentLine.find('#')
            if commentIndex >= 0:
                currentLine = currentLine[:commentIndex]
            tokens = currentLine.split()
            sampledAction = tokens[0]
            sampledLoss = float(tokens[1])
            sampledPropensity = float(tokens[2])
            
            newInstance = None
            sampledY = None
            instanceFeature = None
            if instance_type == 'MultiClass':
                newInstance = Instance.MultiClass(numLabels, numFeatures)
                sampledY = int(sampledAction)
                instanceFeature = scipy.sparse.csr_matrix((1,numFeatures), dtype = numpy.longdouble)
            elif instance_type == 'MultiLabel':
                newInstance = Instance.MultiLabel(numLabels, numFeatures)
                sampledY = numpy.zeros(numLabels, dtype = numpy.int)
                if sampledAction != '-1':
                    for eachLabel in sampledAction.split(','):
                        sampledY[int(eachLabel)] = 1
                instanceFeature = scipy.sparse.csr_matrix((1,numFeatures), dtype = numpy.longdouble)
            elif instance_type == 'Brute':
                newInstance = Instance.Brute(numFeatures)
                sampledY = int(sampledAction)
            
            if instance_type == 'MultiClass' or instance_type == 'MultiLabel':
                for j in range(3, len(tokens)):
                    idVal = tokens[j].split(':')
                    instanceFeature[0,int(idVal[0])] = float(idVal[1])
                
            elif instance_type == 'Brute':
                numActions = int(tokens[3])
                instanceFeature = scipy.sparse.csr_matrix((numActions,numFeatures), dtype = numpy.longdouble)
                for k in range(numActions):
                    currIndex += 1
                    currentAction = allLines[currIndex]
                    commentIndex = currentAction.find('#')
                    if commentIndex >= 0:
                        currentAction = currentAction[:commentIndex]
                    tokens = currentAction.split()
                    currentRow = int(tokens[0])
                    for j in range(1, len(tokens)):
                        idVal = tokens[j].split(':')
                        instanceFeature[currentRow, int(idVal[0])] = float(idVal[1])
            
            newInstance.set(sampledPropensity, sampledLoss, instanceFeature, sampledY)
            instanceList.append(newInstance)
            if i % 20 == 0:
                print(".", flush=True, end='')
        print('')
        print("Dataset:read_bandit_data\t[LOG]\tFinished loading filename: %s Number of instances: %d" %\
                (fileName, numRecords), flush=True)
        return instanceList
        
    def generate_criteo_stream(self, file_name, feature_ids = None):
        if not os.path.exists(file_name):
            print("Dataset:generate_criteo_stream\t[ERR]\tInput file not found at ", file_name, flush=True)
            sys.exit(0)
 
        instances = []

        instanceLines = []

        featureIDs = None
        unseenFeatures = False
        if feature_ids is not None:
            featureIDs = feature_ids
        else:
            featureIDs = collections.Counter()
            unseenFeatures = True
   
        f = None
        if file_name.endswith('.gz'): 
            f = gzip.open(file_name, 'rt')
        else:
            f = open(file_name, 'r')

        line = None
        while True:
            line = next(f, None)
            if line is None:
                break

            line = line.strip()
            if len(line) > 0 and line[0] == 's':
                #Process the previous example
                if len(instanceLines) > 0:
                    numCandidates = len(instanceLines) - 2      #One header, and one trailing empty line
                    header = instanceLines[0]
                    sharedFeatures = header[header.index('|')+2:]
                    sharedCols = []
                    sharedVals = []

                    tokens = sharedFeatures.split(' ')
                    for token in tokens:
                        val = 1
                        featIDStr = token
                        if ':' in token:
                            tempToken = token.split(':', 1)
                            featIDStr = tempToken[0]
                            val = int(tempToken[1])
                        
                        featID = None
                        if featIDStr in featureIDs:
                            featID = featureIDs[featIDStr]
                        elif unseenFeatures:
                            featID = len(featureIDs)
                            featureIDs[featIDStr] = featID
                        else:
                            continue

                        sharedCols.append(featID)
                        sharedVals.append(val)

                    selectedLine = instanceLines[1]
                    splits = selectedLine.split('|', 1)
                    labelInfo = splits[0]
                    toks = labelInfo.split(':',2)
                    loss = float(toks[1])
                    propensity = float(toks[2])
                    chosenAction = 0

                    numSharedCols = len(sharedCols)
                    rows = []
                    cols = []
                    vals = []

                    for j in range(numCandidates):
                        featLine = None
                        if j == 0:
                            featLine = splits[1].split(' ')
                        else:
                            featLine = instanceLines[j+1].split(' ')

                        rows.extend([j] * numSharedCols)
                        cols.extend(sharedCols)
                        vals.extend(sharedVals)

                        for k in range(1, len(featLine)):
                            val = 1
                            featIDStr = featLine[k]
                            if ':' in featIDStr:
                                tempToken = featIDStr.split(':', 1)
                                featIDStr = tempToken[0]
                                val = int(tempToken[1])
                        
                            featID = None
                            if featIDStr in featureIDs:
                                featID = featureIDs[featIDStr]
                            elif unseenFeatures:
                                featID = len(featureIDs)
                                featureIDs[featIDStr] = featID
                            else:
                                continue

                            rows.append(j)
                            cols.append(featID)
                            vals.append(val)

                    currInstance = Instance.Brute(73989)
                    x = scipy.sparse.coo_matrix((vals, (rows, cols)), shape = (numCandidates, 73989), dtype = numpy.int)
                    x = x.tocsr()
                    currInstance.set(propensity, loss, x, 0)
                    instances.append(currInstance)

                    if len(instances) % 10000 == 0:
                        print('.', end='', flush=True)
                        
                    instanceLines.clear()
               
            instanceLines.append(line)

        f.close()

        #Process the final example
        if len(instanceLines) > 0:
            numCandidates = len(instanceLines) - 2      #One header, and one trailing empty line
            header = instanceLines[0]
            sharedFeatures = header[header.index('|')+2:]
            sharedCols = []
            sharedVals = []

            tokens = sharedFeatures.split(' ')
            for token in tokens:
                val = 1
                featIDStr = token
                if ':' in token:
                    tempToken = token.split(':', 1)
                    featIDStr = tempToken[0]
                    val = int(tempToken[1])
                        
                featID = None
                if featIDStr in featureIDs:
                    featID = featureIDs[featIDStr]
                elif unseenFeatures:
                    featID = len(featureIDs)
                    featureIDs[featIDStr] = featID
                else:
                    continue

                sharedCols.append(featID)
                sharedVals.append(val)

            selectedLine = instanceLines[1]
            splits = selectedLine.split('|', 1)
            labelInfo = splits[0]
            toks = labelInfo.split(':',2)
            loss = float(toks[1])
            propensity = float(toks[2])
            chosenAction = 0

            numSharedCols = len(sharedCols)
            rows = []
            cols = []
            vals = []

            for j in range(numCandidates):
                featLine = None
                if j == 0:
                    featLine = splits[1].split(' ')
                else:
                    featLine = instanceLines[j+1].split(' ')

                rows.extend([j] * numSharedCols)
                cols.extend(sharedCols)
                vals.extend(sharedVals)

                for k in range(1, len(featLine)):
                    val = 1
                    featIDStr = featLine[k]
                    if ':' in featIDStr:
                        tempToken = featIDStr.split(':', 1)
                        featIDStr = tempToken[0]
                        val = int(tempToken[1])
                        
                    featID = None
                    if featIDStr in featureIDs:
                        featID = featureIDs[featIDStr]
                    elif unseenFeatures:
                        featID = len(featureIDs)
                        featureIDs[featIDStr] = featID
                    else:
                        continue

                    rows.append(j)
                    cols.append(featID)
                    vals.append(val)
                    
            currInstance = Instance.Brute(73989)
            x = scipy.sparse.coo_matrix((vals, (rows, cols)), shape = (numCandidates, 73989), dtype = numpy.int)
            x = x.tocsr()
            currInstance.set(propensity, loss, x, 0)
            instances.append(currInstance)
        
        return instances, featureIDs
 

        
if __name__ == "__main__":
    seed = 387
    """ 
    d1 = Dataset()
    features, labels = d1.create_synthetic_data(50, 4, 3)
    d1.write_bandit_data('./', features, labels, 'MultiClass', seed)
    d1.write_bandit_data('./', features, labels, 'MultiLabel', seed)
    d1.write_bandit_data('./', features, labels, 'Brute', seed)
    d1.write_bandit_data('./', features, labels, 'PlackettLuce', seed)
    
    d2 = Dataset()
    features, labels = d2.read_supervised_data('./')
    d2.write_bandit_data('./', features, labels, 'MultiClass', seed)
    d2.write_bandit_data('./', features, labels, 'MultiLabel', seed)
    d2.write_bandit_data('./', features, labels, 'Brute', seed)
    
    d3 = Dataset()
    a = d3.read_bandit_data('./', 'MultiClass')
    b = d3.read_bandit_data('./', 'MultiLabel')
    c = d3.read_bandit_data('./', 'Brute')
    """
    d = Dataset()
    instances, featIDs = d.generate_criteo_stream('../Criteo/vw_train.gz')
    #instances, featIDs = d.generate_criteo_stream('/media/adith/DATA/Criteo/temp_train.gz')
    print(len(instances), len(featIDs))
