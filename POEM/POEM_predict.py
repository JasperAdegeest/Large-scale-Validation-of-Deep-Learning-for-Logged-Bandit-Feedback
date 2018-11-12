if __name__ == "__main__":
    import argparse
    import numpy
    import os
    import sys
    import Dataset
    import gzip
    import scipy.sparse
    import pickle
    
    
    parser = argparse.ArgumentParser(description='POEM: Policy Optimizer for Exponential Models.')
    parser.add_argument('--inputFile', '-i', metavar='I', type=str, required=True,
                        help='Testing data file')
    parser.add_argument('--modelFile', '-m', metavar='M', type=str, required=True,
                        help='Model file')
    parser.add_argument('--featureFile', '-f', metavar='F', type=str, required=True,
                        help='Feature file')
    parser.add_argument('--outputFile', '-o', metavar='O', type=str, required=True,
                        help='Predictions output file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.inputFile):
        print("POEM_predict:main\t[ERR]\tPlease provide valid input file via -i", flush=True)
        sys.exit(0)
    
    if not os.path.exists(args.modelFile):
        print("POEM_predict:main\t[ERR]\tPlease provide model file via -m", flush=True)
        sys.exit(0)
 
    if not os.path.exists(args.featureFile):
        print("POEM_predict:main\t[ERR]\tPlease provide feature file via -f", flush=True)
        sys.exit(0)
           
    result = numpy.load(args.modelFile)
    weights = result['arr_0']

    featureFile = open(args.featureFile, 'rb')
    featureIDs = pickle.load(featureFile)
    featureFile.close()
        
    d = Dataset.Dataset()
    testInstances, newFeatIDs = d.generate_criteo_stream(args.inputFile, featureIDs)
 
    f = None
    if args.outputFile.endswith('.gz'):
        f = gzip.open(args.outputFile, 'wt')
    else:
        f = open(args.outputFile, 'w')


    estimatedRisk = 0.0
    for instance in testInstances:
        risk, grad = instance.risk_gradient(weights, -1, False)
        estimatedRisk += risk

        scores = instance.x.dot(weights)
        numCandidates = numpy.shape(instance.x)[0]
        scores = -scores.ravel()
        
        sortIndices = numpy.argsort(scores, axis = None)
        outStr = ''
        for j in range(numCandidates):
            outStr += str(sortIndices[j])+':'+str(scores[sortIndices[j]])+','

        outStr = outStr[:-1] + '\n\n'

        f.write(outStr)
    f.close()

    estimatedRisk /= len(testInstances)
    print("POEM_predict:main\t[LOG]\tPerformance: ", estimatedRisk, flush=True)
    result.close()
 
