import numpy
import sys


class TrainingSet:
    def __init__(self, instances):
        self.instances = instances
        numInstances = len(self.instances)
        sampleWeights = numpy.zeros(numInstances, dtype = numpy.longdouble)
        for i in range(numInstances):
            sampleWeights[i] = numpy.abs(self.instances[i].loss * numpy.exp(self.instances[i].invLogPropensity))
            
        self.sortIndices = numpy.argsort(-sampleWeights, axis = None)
        print("TrainingSet:init\t[LOG]\tNumInstances: %d |Sample weight| [min, max, mean]:" % numInstances,
                sampleWeights.min(), sampleWeights.max(), sampleWeights.mean(), flush=True)
        
        self.meanConstant = None
        self.sqConstant = None
        self.cConstant = None

        self.trainIndices = None
        
    def shuffle(self, mini_batch):
        numInstances = len(self.instances)
        
        partitionSize = int(numInstances * 1.0 / mini_batch)
        perPartitionElements = []
        for i in range(mini_batch):
            currList = None
            if i < (mini_batch - 1):
                currList = self.sortIndices[i*partitionSize:(i+1)*partitionSize].copy()
            else:
                currList = self.sortIndices[i*partitionSize:].copy()
        
            numpy.random.shuffle(currList)        
            perPartitionElements.append(currList.tolist())
            
        shuffledOrder = []
        currIndex = 0
        while len(shuffledOrder) < numInstances:
            currList = perPartitionElements[currIndex]
            currIndex += 1
            if currIndex >= mini_batch:
                currIndex = 0
            if len(currList) <= 0:
                continue
            chosenElement = currList.pop()
            shuffledOrder.append(chosenElement)
        
        self.trainIndices = shuffledOrder
            
    def compute_constants(self, weights, clip_value, var_penalty):
        if var_penalty <= 0:
            self.meanConstant = 1.0
            self.sqConstant = 0.0
            return
            
        numInstances = len(self.instances)
        estimatedRisks = numpy.zeros(numInstances, dtype = numpy.longdouble)
        for i in range(numInstances):
            instance = self.instances[i]
            risk, grad = instance.risk_gradient(weights, clip_value, False)
            estimatedRisks[i] = risk
            
        stdRisk = estimatedRisks.std(dtype = numpy.longdouble, ddof = 1)
        meanRisk = estimatedRisks.mean(dtype = numpy.longdouble)
        
        self.meanConstant = 1 - var_penalty * numpy.sqrt(numInstances) * meanRisk / ((numInstances - 1)*stdRisk)
        self.sqConstant = var_penalty * numpy.sqrt(numInstances) / (2 * (numInstances - 1) * stdRisk)
        print("POEM_learn:compute_constants\t[LOG]\tComputed constants as: [Mean/Sq] \t ", self.meanConstant, self.sqConstant, flush=True)
    
    def update(self, weights, batch_id, batch_size, clip_value, l2_penalty, adagrad_divider):
        numInstances = len(self.instances)
        numBatches = int(numInstances * 1.0 / batch_size)
        currIndices = None
        if batch_id < (numBatches - 1):
            currIndices = self.trainIndices[batch_id*batch_size: (batch_id+1)*batch_size]
        else:
            currIndices = self.trainIndices[batch_id*batch_size:]
    
        gradient = None
        estimatedRisk = 0.0
        for ind in currIndices:
            instance = self.instances[ind]
            risk, grad = instance.risk_gradient(weights, clip_value, True)
            estimatedRisk += risk
            if gradient is None:
                gradient = grad
            elif grad is not None:
                gradient += grad
                
        estimatedRisk = estimatedRisk / numpy.shape(currIndices)[0]

        if gradient is not None:
            gradient = gradient / numpy.shape(currIndices)[0]
            gradient = numpy.divide(gradient, adagrad_divider)
        
            adagrad_divider = numpy.sqrt(numpy.square(adagrad_divider) + numpy.square(gradient))
        
            updateDirection = l2_penalty * weights + (self.meanConstant + self.sqConstant * 2 * estimatedRisk) * gradient
        else:
            updateDirection = l2_penalty * weights

        return weights - 0.5 * updateDirection, adagrad_divider
        

      
if __name__ == "__main__":
    import argparse
    import os
    import Dataset
    import pickle
    
    parser = argparse.ArgumentParser(description='POEM: Policy Optimizer for Exponential Models.')
    parser.add_argument('--inputFile', '-i', metavar='I', type=str, required=True,
                        help='Training data file')
    parser.add_argument('--outputFile', '-o', metavar='O', type=str, required=True,
                        help='Model output file')
    parser.add_argument('--clip', '-c', metavar='C', type=float,
                        help='Clipping hyper-parameter', default=0.97)
    parser.add_argument('--l2', '-l', metavar='L', type=float,
                        help='L2 regularization hyper-parameter', default=1e-6)    
    parser.add_argument('--var', '-v', metavar='V', type=float,
                        help='Variance regularization hyper-parameter', default=2.0)    
    parser.add_argument('--norm', '-n', metavar='N', type=float,
                        help='Self-normalization hyper-parameter', default=1.0)  
    parser.add_argument('--minibatch', '-m', metavar='M', type=int,
                        help='Minibatch size', default=1000)  
    parser.add_argument('--seed', '-s', metavar='S', type=int,
                        help='Random number seed', default=387)
    
    args = parser.parse_args()
    
    numpy.random.seed(args.seed)
    
    if not os.path.exists(args.inputFile):
        print("POEM_learn:main\t[ERR]\tPlease provide valid input file via -i", flush=True)
        sys.exit(0)
        
    d = Dataset.Dataset()
    trainInstances, featureDict = d.generate_criteo_stream(args.inputFile)
    weights = trainInstances[0].parametrize()
    
    numInstances = len(trainInstances)
    losses = numpy.zeros(numInstances, dtype = numpy.longdouble)
    for i in range(numInstances):
        losses[i] = trainInstances[i].loss
        
    #translation = numpy.percentile(losses, args.norm*100, axis = None)
    translation = args.norm
    print("POEM_learn:main\t[LOG]\t[Min,Max,Mean] Loss: ", losses.min(), losses.max(), losses.mean(), flush=True)
    print("POEM_learn:main\t[LOG]\tSelf-normalization fraction and chosen translation: ",
            args.norm, translation, flush=True)
    for i in range(numInstances):
        trainInstances[i].loss = trainInstances[i].loss - translation
    
    clipValue = -1
    if args.clip >= 0:
        propensities = numpy.zeros(numInstances, dtype = numpy.longdouble)
        for i in range(numInstances):
            propensities[i] = numpy.exp(trainInstances[i].invLogPropensity)
    
        clipValue = numpy.percentile(propensities, args.clip*100, axis = None)
        print("POEM_learn:main\t[LOG]\t[Min,Max,Mean] InvPropensity: ", propensities.min(), propensities.max(), propensities.mean(), flush=True)
        print("POEM_learn:main\t[LOG]\tClip percentile and chosen clip constant: ", args.clip, clipValue, flush=True)
    
    trainSet = TrainingSet(trainInstances)
    trainSet.shuffle(args.minibatch)
    
    epochID = 0
    adagradDecay = numpy.ones(weights.shape, dtype = numpy.longdouble)
    
    featureFile = open(args.outputFile+'.features', 'wb')
    pickle.dump(featureDict, featureFile, -1)
    featureFile.close()

    numpy.savez_compressed(args.outputFile+'_'+str(epochID), weights)
    print("POEM_learn:main\t[LOG]\tSaving model", args.outputFile+'_'+str(epochID), "\tNorm of weights:", numpy.linalg.norm(weights), flush=True)
    while True:
        epochID += 1
        print("POEM_learn:main\t[LOG]\tStarting epoch: ", epochID, flush=True)
        #At the start of an epoch, shuffle training set
        trainSet.shuffle(args.minibatch)
        #Also, update the majorization constants
        trainSet.compute_constants(weights, clipValue, args.var)
        
        #Process batches in this epoch
        numTrainInstances = numpy.shape(trainSet.trainIndices)[0]
        numBatches = int(numTrainInstances * 1.0 / args.minibatch)
        for i in range(numBatches):
            weights, adagradDecay = trainSet.update(weights, i, args.minibatch, clipValue, args.l2, adagradDecay)
            
            #If we have processed holdout_period number of batches, time to snapshot weights and update constants
            if (i+1) % 1000 == 0:
                print("POEM_learn:main\t[LOG]\tComputing constants and writing model: ", args.outputFile+'_'+str(epochID)+'_'+str(i+1), "\tNorm of weights:", numpy.linalg.norm(weights), flush=True)
                trainSet.compute_constants(weights, clipValue, args.var)
                
                numpy.savez_compressed(args.outputFile+'_'+str(epochID)+'_'+str(i+1), weights)

        numpy.savez_compressed(args.outputFile+'_'+str(epochID), weights)
        print("POEM_learn:main\t[LOG]\tSaving model", args.outputFile+'_'+str(epochID), "\tNorm of weights:", numpy.linalg.norm(weights), flush=True)

