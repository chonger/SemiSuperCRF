import scipy as SP
import numpy as NP
from scipy.optimize import fmin_bfgs as optimize
from util import SymbolTable,posmax
import itertools

featTable = SymbolTable()
simpleTable = SymbolTable()

class SemiSuperCRF():

    def __init__(self,nParams):
        self.nParams = nParams
        self.params = NP.zeros(nParams)

    '''
    performs one iteration of "EM" on the trees
    first uses BP to calculate marginal probabilities of each feature occurring as "gold"
    then does maximization of exponential model parameters
    '''
    def iteration(self,trees):
    
        '''
        the "gold" counts are done only once, with the inital params
        in this step we calculate (using the provided labels) the expected counts of each feature, summed over all data
        '''
        [t.setPotentials(self.params) for t in trees] #precalculate factor potentials
        [t.sumproduct(True) for t in trees] # do constrainted BP, this gives us expected counts of each feature in fCounts
        goldCounts = NP.zeros(self.nParams)
        for t in trees:
            for x,y in t.fCounts.items():
                goldCounts[x] += y

        '''
        calculates the current gradient using the expected "gold" counts and the argument params
        '''
        def calcGradient(params_tmp): 
            [t.setPotentials(params_tmp) for t in trees]
            [t.sumproduct(False) for t in trees]
            grad = NP.zeros(self.nParams)
            for t in trees:
                for x,y in t.fCounts.items():
                    grad[x] += y
            ret = grad-goldCounts
            #print "Gradient",ret 
            return ret

        '''
        the function to be minimized - 
        the (negative) amount of the potential that is assigned to the current fuzzy "gold" counts
        '''
        def calcF(params_tmp):
            [t.setPotentials(params_tmp) for t in trees]
            [t.sumproduct(False) for t in trees]
            return sum([NP.log(t.norm) for t in trees]) - goldCounts.dot(params_tmp)

        
        #perform the optimization
        self.params = optimize(calcF,self.params,calcGradient)

class BPTree:

    def __init__(self,root):
        self.root = root
        self.norm = 0.0
        self.fCounts = {}

    def getFCount(self,idxs,c):
        def addF(idx,p):
            self.fCounts[idx] = self.fCounts.get(idx,0.0) + p
        p = c / self.norm
        [addF(idx,p) for idx in idxs]

    def sumproduct(self,constrain):
        self.root.inside(constrain)
        self.norm = sum(self.root.insides)
        self.root.outsides = [1.0 for _ in xrange(self.root.nG)]
        self.fCounts = {}
        self.root.outside(self.getFCount,constrain)

    def setPotentials(self,model):
        self.root.computePotentials(model)

    def viterbi(self):
        v = self.root.viterbi_r()
        best = posmax(v,lambda x: x[1])
        return v[best][0]

    def maxprob(self):
        return self.root.maxprob_r()
    

class BPNode:

    '''
    lexFeatures is a single list of simple feature indices
    gFeatures (grounding features) are a list of lists of simple feature indices
    
    goldLabel is None if no supervision is available
    '''
    def __init__(self,lexFeatures,gFeatures,children,goldLabel):
        self.nG = len(gFeatures) # the number of grounding variables
        self.lexFeatures = lexFeatures 
        self.gFeatures = gFeatures
        self.children = children
        self.goldLabel = goldLabel

        self.fullFeats = {} # maps assignment indexes to their list of full feature indexes.
        self.potentials = {}
        self.insides = []
        self.outsides = []

        ranges = [range(c.nG) for c in self.children]
        ranges.insert(0,range(self.nG))

        for x in itertools.product(*ranges):
            self.findFeatures(x) 

    '''
    for a configuration of groundings, precompute all the features
    should get called once, during node initialization
    '''
    def findFeatures(self,idxs):
        assert(len(idxs) == len(self.children) + 1)
        feats = [self.children[ii].gFeatures[idxs[1:][ii]] for ii in xrange(len(self.children))]
        feats.append(self.gFeatures[idxs[0]])
        feats.append(self.lexFeatures)
    
        #cross product all the feature lists
        self.fullFeats[idxs] = [featTable.addID(x) for x in itertools.product(*feats)]
        

    '''
    factor potentials using current model params
    needs to be called after each update to model
    '''
    def computeFactorPotential(self,model,feats):
        return NP.exp(sum([model[x] for x in feats]))

    def computePotentials(self,model):
        for a,b in self.fullFeats.items():
            self.potentials[a] = self.computeFactorPotential(model,b)
            
        for c in self.children:
            c.computePotentials(model)


    '''
    Calculate "inside probs" for each grounding
    this is the sum over all factor assignments, and is the upwards message

    first compute all messages from this nodes factor,
    then do the sum to precompute this nodes upwards message
    
    '''
    def inside(self,constrain):
        [c.inside(constrain) for c in self.children] #recurse first
        
        self.insides = [0.0 for _ in range(self.nG)]
        
        def updateInside(idxs,fp):
            if((not constrain) or self.goldLabel == None or idxs[0] == self.goldLabel): 
                ret = fp
                for i,c in zip(idxs[1:],self.children):
                    ret *= c.insides[i]
                self.insides[idxs[0]] += ret
        
        [updateInside(a,b) for a,b in self.potentials.items()]

    '''
    Calculate "outside probs" for each grounding

    during training, the important part is to run the resultF function,
    which records the expected count of each feature for this datum
    '''
    def outside(self,resultF,constrain):

        for c in self.children:
            c.outsides = [0.0 for _ in range(c.nG)]

        def updateOutside(idxs,fp):
            if((not constrain) or self.goldLabel == None or idxs[0] == self.goldLabel): 
                fullP = fp * self.outsides[idxs[0]]
                for i,c in zip(idxs[1:],self.children):
                    fullP *= c.insides[i]
                #this fullP is probably just the thing we need when calculating the gradient - keep it!
                resultF(self.fullFeats[idxs],fullP)
            
                def updateInner(cI):
                    c = self.children[cI]
                    targ = idxs[cI-1]
                    c.outsides[targ] += fullP / c.insides[targ]

                [updateInner(cI) for cI in xrange(len(self.children))]
                
        [updateOutside(a,b) for a,b in self.potentials.items()]

        for c in self.children:
            c.outside(resultF,constrain)
                
        
    '''
    recursive helper function for viterbi
    '''
    def viterbi_r(self):
        v_c = [x.viterbi_r() for x in self.children]

        v = [[[],0.0] for _ in range(self.nG)]
        
        def updateViterbi(idxs,fp):
            ret = fp
            kids = []
            for i,c in zip(idxs[1:],v_c):
                t = v_c[0]
                p = v_c[1]
                ret *= p
                kids.append(t)

            cur = v[idxs[0]]
            if(cur[1] < ret):
                if(len(kids) == 0):
                    v[idxs[0]] = [[idxs[0]],ret]
                else:
                    v[idxs[0]] = [[idxs[0],kids],ret]
        
        [updateViterbi(a,b) for a,b in self.potentials.items()]

        return v

    '''
    recursive helper funtion for maxprob
    '''
    def maxprob_r(self):
        m_c = [x.maxprob_r() for x in self.children]
        probs = [self.insides[i]*self.outsides[i] for i in range(self.nG)]
        def posmax(seq, key=lambda x:x): 
            return max(enumerate(seq), key=lambda k: key(k[1]))[0]
        best = posmax(probs)
        if(len(m_c) == 0):
            return [best]
        else:
            return [best,m_c]
        
