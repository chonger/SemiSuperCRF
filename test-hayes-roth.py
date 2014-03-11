from model import *
from util import posmax

def main():

    data = []

    labels = []
    labels.append([simpleTable.addID("c0")])
    labels.append([simpleTable.addID("c1")])
    labels.append([simpleTable.addID("c2")])
    
    for line in file("hayes-roth.data.txt"):
        p = line.strip().split(",")
        if(len(p) > 1):
            cl = int(p[5])-1
            fs = ["F" + str(a) + "-" + str(b) for a,b in zip(range(4),p[1:5])]
            fs = [simpleTable.addID(x) for x in fs]
            
            n = BPNode(fs,labels,[],cl)
            t = BPTree(n)

            data.append(t)    


    print "Data size   :",len(data)
    print "num feats   :",featTable.size()
    print "num sfeats  :",simpleTable.size()
    
    ssCRF = SemiSuperCRF(featTable.size())

    #for fully supervised problems only one iteration is necessary
    ssCRF.iteration(data[:100])
    
    acc = 0.0
    tot = 0.0
    for d in data[100:]:

        '''
        these are required for all evals
        '''
        d.setPotentials(ssCRF.params)
        d.sumproduct(False)

        '''
        grab the highest inside prob of root directly
        '''
        predict = posmax(d.root.insides)
        if(predict == d.root.goldLabel):
            acc += 1
        tot += 1

        '''
        use viterbi eval code
        '''
        assert(predict == d.viterbi()[0])

        '''
        use max prob eval code
        '''
        assert(predict == d.maxprob()[0])

    print "Classification Accuracy : ",acc/tot

if __name__ == "__main__":
    main()
