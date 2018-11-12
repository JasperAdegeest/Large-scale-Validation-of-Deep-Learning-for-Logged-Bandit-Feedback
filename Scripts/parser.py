from collections import deque
import numpy
import sys
import gzip
import scipy.misc
import math

#Expected Cmdline: python parser.py [criteo_data_file] [vw_output_prefix] <compressed_output?> <click_encoding> <no-click_encoding>
#Defaults for optional args:
#   <compressed_output?>    False
#   <click_encoding>        0.001
#   <no-click_encoding>     0.999
#Typical usage:
#python parser.py CriteoBannerFillingChallenge.txt.gz vw_compressed c
#python parser.py CriteoBannerFillingChallenge.txt.gz vw_raw


def gzipOrNot(filename):
    f = None
    if filename.endswith('.gz'):
        f = gzip.open(filename, 'rt')
    else:
        f = open(filename, 'r')

    return f

if len(sys.argv) < 3:
    print("Parser:main \t [ERR] \t Expected Cmdline:  \
                python parser.py [criteo_data_file] [vw_output_prefix] <compressed_output?>",
                flush=True)
    sys.exit(0)

fileName = sys.argv[1]
f = gzipOrNot(fileName)

compressed = False
if len(sys.argv) > 3:
    compressed = (sys.argv[3][0] == 'c')

posLoss = 0.001
if len(sys.argv) > 4:
    posLoss = float(sys.argv[4])

negLoss = 0.999
if len(sys.argv) > 5:
    negLoss = float(sys.argv[5])

ofTrain = None
ofValidate = None
ofTest = None
outputPrefix = sys.argv[2]
if compressed:
    ofTrain = gzip.open(outputPrefix+'_train.gz', 'wt')
    ofValidate = gzip.open(outputPrefix+'_validate.gz', 'wt')
    ofTest = gzip.open(outputPrefix+'_test.gz', 'wt')
else:
    ofTrain = open(outputPrefix+'_train','w')
    ofValidate = open(outputPrefix+'_val','w')
    ofTest = open(outputPrefix+'_test','w')

#Maintain Sanity checks for each k-slot banner type, k=1...6
#Min/Max/Mean |Loss / Propensity| -- Loss encoded as: click = 0.001, no-click = 0.999
minEstimate = -numpy.ones(6, dtype = numpy.longdouble)
maxEstimate = numpy.zeros(6, dtype = numpy.longdouble)
numPosInstances = numpy.zeros(6, dtype = numpy.int)
numNegInstances = numpy.zeros(6, dtype = numpy.int)
avgEstimate = numpy.zeros(6, dtype = numpy.longdouble)
#Min/Max/Mean (1{click} / Propensity)
minLabelEstimate = -numpy.ones(6, dtype = numpy.longdouble)
maxLabelEstimate = numpy.zeros(6, dtype = numpy.longdouble)
avgLabelEstimate = numpy.zeros(6, dtype = numpy.longdouble)
#Min/Max/Mean (1 / Propensity)
minPropensity = -numpy.ones(6, dtype = numpy.longdouble)
maxPropensity = numpy.zeros(6, dtype = numpy.longdouble)
avgPropensity = numpy.zeros(6, dtype = numpy.longdouble)

#Use epsilon-logger policies for policy-specific imp. wt. checks
#epsilon = numpy.linspace(0, 1, num=21, endpoint = True, dtype = numpy.longdouble)
epsilon = numpy.array([0, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875, 0.99609375, 0.998046875, 0.999023438, 1], dtype = numpy.longdouble) 
epsilonComplement = 1 - epsilon
numEpsilons = numpy.shape(epsilon)[0]
#Importance Weights, Mean and Var: Running Avg
denominatorM1 = numpy.zeros((6, numEpsilons), dtype = numpy.longdouble)
denominatorM2 = numpy.zeros((6, numEpsilons), dtype = numpy.longdouble)
brokenDenominatorM1 = numpy.zeros((6, numEpsilons), dtype = numpy.longdouble)
brokenDenominatorM2 = numpy.zeros((6, numEpsilons), dtype = numpy.longdouble)
#Importance-weighted Estimates, Mean and Var: Running Avg
numeratorM1 = numpy.zeros((6, numEpsilons), dtype = numpy.longdouble)
numeratorM2 = numpy.zeros((6, numEpsilons), dtype = numpy.longdouble)

header = True
numRemainingCandidates = 0
save = None
train = 0                           #   0: Test; 1: Train; 2: Validate     33-33-33% split
label = None
propensity = None

linesProcessed = 0
for line in f:
    linesProcessed += 1
    tokens = line.split(' ')
    if header:
        #Expect a "example" line
        header = False
        numRemainingCandidates = int(tokens[6])
        label = int(tokens[3])
        loss = negLoss
        if label == 1:
            loss = posLoss

        propensity = float(tokens[4])
        lossPropensity = loss / propensity
        labelPropensity = label / propensity
        invPropensity = 1.0 / propensity

        numSlots = int(tokens[5]) - 1
        if numSlots == 0:
            save = True
            train = (train + 1) % 3
        else:
            save = False

        #Update sanity check numbers
        if minEstimate[numSlots] < 0 or minEstimate[numSlots] > lossPropensity:
            minEstimate[numSlots] = lossPropensity

        if maxEstimate[numSlots] < lossPropensity:
            maxEstimate[numSlots] = lossPropensity

        if minLabelEstimate[numSlots] < 0 or minLabelEstimate[numSlots] > labelPropensity:
            minLabelEstimate[numSlots] = labelPropensity

        if maxLabelEstimate[numSlots] < labelPropensity:
            maxLabelEstimate[numSlots] = labelPropensity

        if minPropensity[numSlots] < 0 or minPropensity[numSlots] > invPropensity:
            minPropensity[numSlots] = invPropensity

        if maxPropensity[numSlots] < invPropensity:
            maxPropensity[numSlots] = invPropensity

        numNegInstances[numSlots] += 1
        if label == 1:
            numNegInstances[numSlots] -= 1
            numPosInstances[numSlots] += 1

        n = numNegInstances[numSlots] + numPosInstances[numSlots]

        delta = lossPropensity - avgEstimate[numSlots]
        avgEstimate[numSlots] += (delta / n)
        
        delta = labelPropensity - avgLabelEstimate[numSlots]
        avgLabelEstimate[numSlots] += (delta / n)

        delta = invPropensity - avgPropensity[numSlots]
        avgPropensity[numSlots] += (delta / n)

        #Probability for epsilon-logger to pick displayed action: epsilon * propensity + (1 - epsilon) / numActions
        propensityNumActions = propensity * scipy.misc.comb(numRemainingCandidates, numSlots+1) * math.factorial(numSlots+1)
        newPolicyWeight = epsilon + (epsilonComplement / propensityNumActions)
        delta = newPolicyWeight - brokenDenominatorM1[numSlots,:]
        brokenDenominatorM1[numSlots,:] += (delta / n)
        brokenDenominatorM2[numSlots,:] += numpy.multiply(delta, newPolicyWeight - brokenDenominatorM1[numSlots,:])

        newPolicyWeight *= 10
        if label == 1:
            newPolicyWeight /= 10

        newPolicyEstimate = label * newPolicyWeight

        delta = newPolicyWeight - denominatorM1[numSlots,:]
        denominatorM1[numSlots,:] += (delta / n)
        denominatorM2[numSlots,:] += numpy.multiply(delta, newPolicyWeight - denominatorM1[numSlots,:])

        delta = newPolicyEstimate - numeratorM1[numSlots,:]
        numeratorM1[numSlots,:] += (delta / n)
        numeratorM2[numSlots,:] += numpy.multiply(delta, newPolicyEstimate - numeratorM1[numSlots,:])

        #Apply the sub-sampling factor to the propensities, so subsequent processing steps can be impervious to it
        propensity /= 10
        if label == 1:
            propensity *= 10

        if save:
            tag = tokens[1][:-1]
            feats = tokens[7]+' '+tokens[8]
            for j in range(9, len(tokens)):
                feats += ' ' + tokens[j].replace(':','_')
            outLine = 'shared '+tag+'| '+ feats

            outFile = None
            if train == 1:
                outFile = ofTrain
            elif train == 2:
                outFile = ofValidate
            else:
                outFile = ofTest

            outFile.write(outLine)

    else:
        #Expect a "exid" line
        numRemainingCandidates -= 1

        if save:
            outFile = None
            if train == 1:
                outFile = ofTrain
            elif train == 2:
                outFile = ofValidate
            else:
                outFile = ofTest

            #Output to file
            outLine = ''
            if label is not None:
                outLabel = str(negLoss)
                if label == 1:
                    outLabel = str(posLoss)
                outLine = '0:'+outLabel+':'+str(propensity) + ' '
                label = None
                propensity = None
            
            outLine += '|'
            featTokens = deque(tokens[2:])
            
            while len(featTokens) > 0:
                currFeat = featTokens.popleft()
                currCount = 1
                while (len(featTokens) > 0) and (featTokens[0] == currFeat):
                    featTokens.popleft()
                    currCount += 1

                if currCount > 1:
                    outLine += ' ' + currFeat.replace(':','_') + ':' + str(currCount)
                else:
                    outLine += ' ' + currFeat.replace(':','_')
            
            outFile.write(outLine)

            if numRemainingCandidates == 0:
                outFile.write('\n')

        if numRemainingCandidates == 0:
            header = True
            save = None
            label = None
            propensity = None

    if linesProcessed % 200000 == 0:
        print('.', end='', flush=True)

ofTrain.close()
ofValidate.close()
ofTest.close()
f.close()

modifiedDenom = numPosInstances + 10*numNegInstances
originalDenom = numPosInstances + numNegInstances
print('', flush=True)
print("Num[Pos/Neg]Instances for k-slot banners: \n", numPosInstances, "\n", numNegInstances, flush=True)
print("Num[Original(est)/Subsampled] instances: \n", modifiedDenom, "\n", originalDenom, flush=True)

print("[Min/Max/Mean] Loss * InvPropensity for k-slot banners: \n", 
        minEstimate, "\n", maxEstimate, "\n", avgEstimate, flush=True)
print("[Min/Max/Mean] Label * InvPropensity for k-slot banners: \n", 
        minLabelEstimate, "\n", maxLabelEstimate, "\n", avgLabelEstimate, flush=True)
print("[Min/Max/Mean] InvPropensity for k-slot banners: \n", 
        minPropensity, "\n", maxPropensity, "\n", avgPropensity, flush=True)

def compute_result(originalInstances, subsampledInstances, estimateM1, estimateM2, weightM1, weightM2):
    IPS = estimateM1 * subsampledInstances / originalInstances
    #estimateM2 / subsampledInstances gives Var(RV). We want sqrt(Var(RV)) / sqrt(subsampledInstances)
    #and then scale this by the scaling constant subsampledInstances / originalInstances. 
    IPS_std = 2.58 * numpy.sqrt(estimateM2) / originalInstances

    ImpWt = weightM1 * subsampledInstances / originalInstances
    ImpWt_std = 2.58 * numpy.sqrt(weightM2) / originalInstances

    SNIPS = numpy.divide(estimateM1, weightM1)

    print("IPS(*10^4) : \t ", IPS*1e4, flush=True)
    print("StdErr(IPS)*10^4 : \t ", IPS_std*1e4, flush=True)

    print("SN-IPS(*10^4) : \t ", SNIPS*1e4, flush=True)
    
    print("AvgImpWt : \t ", ImpWt, flush=True)
    print("StdErr(AvgImpWt) : \t ", ImpWt_std, flush=True)


for slots in range(6):
    print("NumSlots: \t ", slots + 1, flush=True)
    compute_result(modifiedDenom[slots], originalDenom[slots], 
                    numeratorM1[slots,:], numeratorM2[slots,:],
                    denominatorM1[slots,:], denominatorM2[slots,:])

    print("BrokenImpWt : \t ", brokenDenominatorM1[slots,:], flush=True)
    broken_std = 2.58 * numpy.sqrt(brokenDenominatorM2[slots,:]) / originalDenom[slots]
    print("StdErr(BrokenImpWt) : \t ", broken_std, flush=True)

