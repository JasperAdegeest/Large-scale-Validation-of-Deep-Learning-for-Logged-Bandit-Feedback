import sys
import numpy
from itertools import (takewhile, repeat)
import gzip


#Expected Cmdline: python scorer.py [vw_prediction_file] [vw_input_file] [identifier_for_negative_label]
#Typical usage: 
#python scorer.py model_test_predictions vw_test.gz 0.999


def gzipOrNot(filename, mode):
    f = None
    if mode != 'b' and mode != 't':
        print("Scorer:gzipOrNot \t [ERR] \t Expected filemodes: r/t", flush=True)
        sys.exit(0)

    if filename.endswith('.gz'):
        if mode == 'b':
            f = gzip.open(filename, 'rb')
        else:
            f = gzip.open(filename, 'rt')
    else:
        if mode == 'b':
            f = open(filename, 'rb')
        else:
            f = open(filename, 'r')

    return f


#To get how many instances there are (initializing per-instance estimates)
#Use number of lines in vw_prediction_file
def rawincount(filename):
    f = gzipOrNot(filename, 'b')
    bufgen = takewhile(lambda x: x, (f.read(1024*1024) for _ in repeat(None)))
    lines = sum( buf.count(b'\n') for buf in bufgen )
    f.close()
    return lines


if len(sys.argv) < 4:
    print("Scorer:main \t [ERR] \t Expected Cmdline:  \
                python scorer.py [vw_prediction_file] [vw_input_file] [identifier_for_negative_label]",
                flush=True)
    sys.exit(0)


predictionsFile = sys.argv[1]
testFile = sys.argv[2]
negLabel = float(sys.argv[3])

numLines = rawincount(predictionsFile)
maxInstances = int(numLines / 2)             #Account for empty \n that vw predictions are padded with

inpFile = gzipOrNot(predictionsFile, 't')
dataFile = gzipOrNot(testFile, 't')

numPosInstances = 0
numNegInstances = 0

#Random
randNumerator = numpy.zeros(maxInstances, dtype = numpy.float)
randDenominator = numpy.zeros(maxInstances, dtype = numpy.float)

#Logger
logNumerator = numpy.zeros(maxInstances, dtype = numpy.float)
logDenominator = numpy.zeros(maxInstances, dtype = numpy.float)

#NewPolicy
predictionNumerator = numpy.zeros(maxInstances, dtype = numpy.float)
predictionDenominator = numpy.zeros(maxInstances, dtype = numpy.float)

#NewPolicy - Stochastic
predictionStochasticNumerator = numpy.zeros(maxInstances, dtype = numpy.float)
predictionStochasticDenominator = numpy.zeros(maxInstances, dtype = numpy.float)

exampleLine = None

currID = -1
for line in inpFile:
    strippedLine = line.strip()
    if strippedLine == '':
        continue
        
    if exampleLine is None:
        exampleLine = next(dataFile)    #Now we have the "shared <blah> info

    exampleLine = next(dataFile)    #Now we have the label/propensity info
    labelProp = exampleLine[2:exampleLine.index('|')]
    numCandidates = 0
    while exampleLine[0] != 's':
        numCandidates += 1
        exampleLine = next(dataFile, None)
        if exampleLine is None:
            break
    
    numCandidates -= 1              #Account for the blank line between examples

    label, propensity = labelProp.split(':', 1) 
    label = float(label)
    propensity = float(propensity)

    rectifiedLabel = 0
    numNegInstances += 1

    if label != negLabel:
        rectifiedLabel = 1
        numPosInstances += 1
        numNegInstances -= 1

    currID += 1

    randWeight = 1.0 / (numCandidates * propensity)
    randNumerator[currID] = rectifiedLabel * randWeight
    randDenominator[currID] = randWeight

    logWeight = 10.0
    if label != negLabel:
        logWeight = 1.0
    logNumerator[currID] = rectifiedLabel * logWeight
    logDenominator[currID] = logWeight

    #Finally parse the predicted scores
    tokens = strippedLine.split(',')
   
    #For deterministic policy 
    bestScore = None
    bestClasses = []

    #For stochastic policy
    scoreLoggedAction = None
    scoreNormalizer = 0.0
    scoreOffset = None                  #Hopefully better stability when doing exp(-score-offset)

    for token in tokens:
        splitToks = token.split(':',1)
        currScore = float(splitToks[1])
        actionToken = splitToks[0]
        if bestScore is None or bestScore == currScore:         
            #This exploits the fact that vw predictions are sorted by score(ascending)
            bestClasses.append(actionToken)
            bestScore = currScore

        if scoreOffset is None:
            scoreOffset = -currScore

        probScore = numpy.exp(-currScore-scoreOffset)
        scoreNormalizer += probScore
        if actionToken == '0':
            scoreLoggedAction = probScore

    if '0' in bestClasses:
        predictionWeight = 1.0 / (len(bestClasses) * propensity)
        predictionNumerator[currID] = rectifiedLabel * predictionWeight
        predictionDenominator[currID] = predictionWeight

    predictionStochasticWeight = 1.0 * scoreLoggedAction / (scoreNormalizer * propensity)
    predictionStochasticNumerator[currID] = rectifiedLabel * predictionStochasticWeight
    predictionStochasticDenominator[currID] = predictionStochasticWeight

    if currID % 50000 == 0:
        print('.', end='', flush=True)

inpFile.close()
dataFile.close()

modifiedDenom = numPosInstances + 10*numNegInstances
scaleFactor = numpy.sqrt(maxInstances) / modifiedDenom

print('', flush=True)
print("Num[Pos/Neg]Test Instances:", numPosInstances, numNegInstances, flush=True)
print("MaxID; currID:", maxInstances, currID, flush=True)
print("Approach & IPS(*10^4) & StdErr(IPS)*10^4 & SN-IPS(*10^4) & StdErr(SN-IPS)*10^4 & AvgImpWt & StdErr(AvgImpWt) \\", flush=True)

def compute_result(approach, numerator, denominator):
    IPS = numerator.sum(dtype = numpy.longdouble) / modifiedDenom
    IPS_std = 2.58 * numerator.std(dtype = numpy.longdouble) * scaleFactor        #99% CI
    ImpWt = denominator.sum(dtype = numpy.longdouble) / modifiedDenom
    ImpWt_std = 2.58 * denominator.std(dtype = numpy.longdouble) * scaleFactor    #99% CI
    SNIPS = IPS / ImpWt

    normalizer = ImpWt * modifiedDenom

    #See Art Owen, Monte Carlo, Chapter 9, Section 9.2, Page 9
    #Delta Method to compute an approximate CI for SN-IPS
    Var = numpy.sum(numpy.square(numerator) +\
                    numpy.square(denominator) * SNIPS * SNIPS -\
                    2 * SNIPS * numpy.multiply(numerator, denominator), dtype = numpy.longdouble) / (normalizer * normalizer)

    SNIPS_std = 2.58 * numpy.sqrt(Var) / numpy.sqrt(maxInstances)                 #99% CI
    print(approach, "&", '%.3f' % (IPS*1e4), "&",  '%.3f' % (IPS_std*1e4), "&", 
                         '%.3f' % (SNIPS*1e4), "&",  '%.3f' % (SNIPS_std*1e4), "&", 
                         '%.3f' % ImpWt, "&",  '%.3f' % ImpWt_std, "\\\\", flush=True)

compute_result('Random', randNumerator, randDenominator)
compute_result('Logger', logNumerator, logDenominator)
compute_result('NewPolicy', predictionNumerator, predictionDenominator)
compute_result('NewPolicy-Stochastic', predictionStochasticNumerator, predictionStochasticDenominator)

