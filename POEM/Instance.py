import numpy
import scipy.misc
import scipy.special
import sys


class Instance:
    def __init__(self, instance_type):
        self.instanceType = instance_type
        self.unset = True

    def set(self, propensity, loss, x, y):
        #x: Expect a sparse matrix
        if propensity <= 0:
            print("Instance:set\t[ERR]\tInvalid propensity ", propensity, flush=True)
            sys.exit(0)
            
        self.invLogPropensity = -numpy.log(propensity)
        self.loss = loss
        self.x = x
        self.y = y
        
        self.unset = False
    

class MultiClass(Instance):
    def __init__(self, num_classes, num_features):
        Instance.__init__(self, 'MultiClass')
        self.numClasses = num_classes
        self.numFeatures = num_features
        #x: Expect sparse csr_matrix of dimensions of (1, num_features)
        #y: Expect int, 0 <= y < num_classes

    def parametrize(self):
        weights = numpy.zeros((self.numFeatures, self.numClasses), dtype=numpy.longdouble)
        return weights

    def risk_gradient(self, weights, clip, compute_gradient):
        if self.unset:
            print("MultiClass:risk_gradient\t[ERR]\tSet loss, propensity, x, y first", flush=True)
            sys.exit(0)
        
        scores = self.x.dot(weights)
        scores = scores.ravel()
        partition = scipy.misc.logsumexp(scores)

        logProbability = scores[self.y] - partition
        importanceWeight = numpy.exp(logProbability + self.invLogPropensity)

        clipped = False
        if (clip > 0) and (importanceWeight > clip):
            importanceWeight = clip
            clipped = True

        risk = self.loss * importanceWeight
        gradient = None
        if (not clipped) and compute_gradient:
            probabilityPerClass = numpy.exp(scores - partition)
            gradient = numpy.zeros(weights.shape, dtype = numpy.longdouble)
            rowIndices = self.x.indices[self.x.indptr[0]:self.x.indptr[1]]
            for i in range(self.numClasses):
                if i != self.y:
                    gradient[rowIndices, i] = -risk * probabilityPerClass[i] * self.x.data[self.x.indptr[0]:self.x.indptr[1]]
                else:
                    gradient[rowIndices, i] = risk * (1 - probabilityPerClass[i]) * self.x.data[self.x.indptr[0]:self.x.indptr[1]]
        return risk, gradient

        
class MultiLabel(Instance):
    def __init__(self, num_labels, num_features):
        Instance.__init__(self, 'MultiLabel')
        self.numLabels = num_labels
        self.numFeatures = num_features
        #x: Expect sparse csr_matrix of dimensions of (1, num_features)
        #y: Expect numpy array of dimensions (num_labels,)

    def parametrize(self):
        weights = numpy.zeros((self.numFeatures, self.numLabels), dtype=numpy.longdouble)
        return weights

    def risk_gradient(self, weights, clip, compute_gradient):
        if self.unset:
            print("MultiLabel:risk_gradient\t[ERR]\tSet loss, propensity, x, y first", flush=True)
            sys.exit(0)
        
        scores = self.x.dot(weights)
        scores = scores.ravel()

        signedY = 2*self.y - 1

        labelScores = numpy.multiply(scores, signedY)
        probabilityPerSeenLabel = scipy.special.expit(labelScores)
        
        zeroMask = probabilityPerSeenLabel <= 0
        probabilityPerSeenLabel[zeroMask] = 1.0
        
        logProbabilityPerSeenLabel = numpy.log(probabilityPerSeenLabel)
        logProbability = numpy.sum(logProbabilityPerSeenLabel, dtype=numpy.longdouble)
        
        importanceWeight = numpy.exp(logProbability + self.invLogPropensity)
        if zeroMask.sum(dtype = numpy.int) > 0:
            importanceWeight = 0
        
        clipped = False
        if (clip > 0) and (importanceWeight > clip):
            importanceWeight = clip
            clipped = True

        risk = self.loss * importanceWeight
        gradient = None
        if (not clipped) and compute_gradient:
            gradient = numpy.zeros(weights.shape, dtype = numpy.longdouble)
            probabilityPerLabel = scipy.special.expit(scores)
            rowIndices = self.x.indices[self.x.indptr[0]:self.x.indptr[1]]
            for i in range(self.numLabels):
                gradient[rowIndices, i] = risk * (self.y[i] - probabilityPerLabel[i]) * self.x.data[self.x.indptr[0]:self.x.indptr[1]]
        return risk, gradient


class PlackettLuce(Instance):
    def __init__(self, num_candidates, num_slots, num_features):
        Instance.__init__(self, 'PlackettLuce')
        if num_slots > num_candidates:
            print("PlackettLuce:init\t[ERR]\tCannot have fewer candidates than slots", flush=True)
            sys.exit(0)
 
        self.numCandidates = num_candidates
        self.numSlots = num_slots
        self.numFeatures = num_features
        #x: Expect sparse csr_matrix of dimensions of (num_candidates, num_features)
        #y: Expect numpy array of dimensions (num_slots,)
        #Each unique element 0 <= y_i < num_candidates
    
    def parametrize(self):
        weights = numpy.zeros(self.numFeatures, dtype=numpy.longdouble)
        return weights
    
    def risk_gradient(self, weights, clip, compute_gradient):
        if self.unset:
            print("PlackettLuce:risk_gradient\t[ERR]\tSet loss, propensity, x, y first", flush=True)
            sys.exit(0)
    
        scores = self.x.dot(weights)
        scores = scores.ravel()
        candidateWeights = numpy.ones(numpy.shape(scores), dtype = numpy.int)
        logProbability = 0.0
        for j in range(self.numSlots):
            currPartition = scipy.misc.logsumexp(scores, b = candidateWeights)
            candidateWeights[self.y[j]] = 0
            logProbability += scores[self.y[j]]
            logProbability -= currPartition

        importanceWeight = numpy.exp(logProbability + self.invLogPropensity)
        
        clipped = False
        if (clip > 0) and (importanceWeight > clip):
            importanceWeight = clip
            clipped = True

        risk = self.loss * importanceWeight
        gradient = None
        if (not clipped) and compute_gradient:
            gradient = numpy.zeros(weights.shape, dtype = numpy.longdouble)
            candidateWeights = numpy.ones(numpy.shape(scores), dtype = numpy.int)
            for i in range(self.numSlots):
                rowIndices = self.x.indices[self.x.indptr[self.y[i]]:self.x.indptr[self.y[i]+1]]
                gradient[rowIndices] += self.x.data[self.x.indptr[self.y[i]]:self.x.indptr[self.y[i]+1]]

                currPartition = scipy.misc.logsumexp(scores, b = candidateWeights)
                currProbabilities = numpy.exp(scores - currPartition)
                currProbabilities[candidateWeights <= 0] = 0
               
                weightedProbabilities = scipy.sparse.spdiags(currProbabilities, 0, self.numCandidates, self.numCandidates) * self.x 
                gradient -= (weightedProbabilities.sum(axis = 0, dtype = numpy.longdouble)).A1

                candidateWeights[self.y[i]] = 0

            gradient *= risk
            return risk, gradient


       
class Brute(Instance):
    def __init__(self, num_features):
        Instance.__init__(self, 'Brute')
        self.numFeatures = num_features
        #x: Expect sparse csr_matrix of dimensions of (num_candidates, num_features)
        #y: Expect int, 0 <= y < num_candidates

    def parametrize(self):
        weights = numpy.zeros(self.numFeatures, dtype=numpy.longdouble)
        return weights

    def risk_gradient(self, weights, clip, compute_gradient):
        if self.unset:
            print("Brute:risk_gradient\t[ERR]\tSet loss, propensity, x, y first", flush=True)
            sys.exit(0)
        
        scores = self.x.dot(weights)
        scores = scores.ravel()
        partition = scipy.misc.logsumexp(scores)

        logProbability = scores[self.y] - partition
        importanceWeight = numpy.exp(logProbability + self.invLogPropensity)

        clipped = False
        if (clip > 0) and (importanceWeight > clip):
            importanceWeight = clip
            clipped = True

        risk = self.loss * importanceWeight
        gradient = None
        if (not clipped) and compute_gradient:
            gradient = numpy.zeros(weights.shape, dtype = numpy.longdouble)
            probabilityPerY = numpy.exp(scores - partition)
            numY = numpy.shape(self.x)[0]

            for i in range(numY):
                rowIndices = self.x.indices[self.x.indptr[i]:self.x.indptr[i+1]]
                gradient[rowIndices] += -risk * probabilityPerY[i] * self.x.data[self.x.indptr[i]:self.x.indptr[i+1]]
            rowIndices = self.x.indices[self.x.indptr[self.y]:self.x.indptr[self.y+1]]
            gradient[rowIndices] += risk * self.x.data[self.x.indptr[self.y]:self.x.indptr[self.y+1]]
        return risk, gradient

        
        
if __name__ == "__main__":
    import scipy.sparse

    a = MultiClass(5, 20)
    a_weights = a.parametrize()
    a_x = numpy.ones((1, 20), dtype = numpy.longdouble)
    a_x = scipy.sparse.csr_matrix(a_x)
    a.set(0.1, 1.0, a_x, 3)
    print("Multiclass risk/gradient", a.risk_gradient(a_weights, 5.0, True))
    
    b = MultiLabel(4, 10)
    b_weights = b.parametrize()
    b_x = numpy.ones((1, 10), dtype = numpy.longdouble)
    b_x = scipy.sparse.csr_matrix(b_x)
    b.set(0.1, 1.0, b_x, numpy.array([0,0,1,1], dtype = numpy.int))
    print("Multilabel risk/gradient", b.risk_gradient(b_weights, 5.0, True))
    
    c = Brute(30)
    c_weights = c.parametrize()
    c_x = numpy.zeros((2,30), dtype = numpy.longdouble)
    c_x[0,0:15] = 1
    c_x[1,15:] = 1
    c_x = scipy.sparse.csr_matrix(c_x)
    c.set(0.1, 1.0, c_x, 1)
    print("Brute risk/gradient", c.risk_gradient(c_weights, 5.0, True))
   
    d = PlackettLuce(5, 3, 8)
    d_weights = d.parametrize()
    d_x = numpy.zeros((5, 8), dtype = numpy.longdouble)
    numpy.fill_diagonal(d_x, 1)
    d_x = scipy.sparse.csr_matrix(d_x)
    d.set(0.1, 1.0, d_x, numpy.array([1,0,2], dtype = numpy.int))
    print("PlackettLuce risk/gradient", d.risk_gradient(d_weights, 50000.0, True))
