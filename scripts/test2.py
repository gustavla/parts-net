from __future__ import division, print_function,absolute_import
import pylab as plt
import amitgroup.plot as gr
import numpy as np
import amitgroup as ag
import os
import pnet
def extract(ims,allLayers):
    print(allLayers)
    curX = ims
    for layer in allLayers:
        curX = layer.extract(curX)
    return curX

def test2():
    X = np.load('test4.npy')
    model = X.item()
    net = pnet.PartsNet.load_from_dict(model)
    allLayer = net.layers
    print(allLayer)
    ims, labels = ag.io.load_mnist('testing')
    extractedParts = extract(ims[0:200],allLayer[0:2])
    #return extractedParts
    allParts = extractedParts[0]
    parts_layer = allLayer[1]
    parts = parts_layer._parts.reshape(50,4,4)
    #for i in range(200):
    ims = ims[0:200]
    labels = labels[0:200]
    #print(ims.shape)
    classifiedLabel = net.classify(ims)
    misclassify = np.nonzero(classifiedLabel!=labels)
    misclassify = np.append([],np.asarray(misclassify, dtype=np.int))
    numMisclassify = len(misclassify)
    image = np.ones((numMisclassify,25 * 5,25*5)) * 0.5
    print(misclassify)
    for j in range(numMisclassify):
        i = int(misclassify[j])
        
        print(allParts[i].shape)
        thisParts = allParts[i].reshape(allParts[i].shape[0:2])
        for m in range(25):
            for n in range(25):
                if(thisParts[m,n]!=-1):
                    image[j,m*5:m*5+4,n*5:n*5+4] = parts[thisParts[m,n]]
                else:
                    image[j,m*5:m*5+4,n*5:n*5+4] = 0

    gr.images(image)

def displayParts():
    X =  np.load('test4.npy')
    model = X.item()
    numParts1 = model['layers'][1]['num_parts']
    numParts2 = model['layers'][3]['num_parts']
    net = pnet.PartsNet.load_from_dict(model)
    allLayer = net.layers
    print(allLayer)
    ims,labels = ag.io.load_mnist('training') 
    extractedFeature = []
    for i in range(5):
        extractedFeature.append(extract(ims[0:1000],allLayer[0:i])[0])
    extractedParts1 = extractedFeature[2]
    #extractedParts11 = extract(ims[0:1000],allLayer[0:6])[1]
    #print(extractedParts11)
    extractedParts2 = extractedFeature[4]
    print(extractedParts2.shape)
    partsPlot1 = np.zeros((numParts1,4,4))
    partsCodedNumber1 = np.zeros(numParts1)
    partsPlot2 = np.zeros((numParts2,6,6))
    partsCodedNumber2 = np.zeros(numParts2)

    for i in range(1000):
        codeParts1 = extractedParts1[i].reshape(extractedParts1[i].shape[0:2])
        codeParts2 = extractedParts2[i].reshape(extractedParts2[i].shape[0:2])
        for m in range(25):
            for n in range(25):
                if(codeParts1[m,n]!=-1):
                    partsPlot1[codeParts1[m,n]]+=ims[i,m:m+4,n:n+4] 
                    partsCodedNumber1[codeParts1[m,n]]+=1
        for p in range(8):
            for q in range(8):
                if(codeParts2[p,q]!=-1):
                    partsPlot2[codeParts2[p,q]]+=ims[i,3 * p:3 * p+6,3 * q:3 * q+6]
                    partsCodedNumber2[codeParts2[p,q]]+=1
                #if(codeParts2[p,q,1]!=-1):
                #    partsPlot2[codeParts2[p,q,1]]+=ims[i,p:p+10,q:q+10]
                #    partsCodedNumber2[codeParts2[p,q,1]]+=1
    for j in range(numParts1):
        partsPlot1[j] = partsPlot1[j]/partsCodedNumber1[j]
    for k in range(numParts2):
        partsPlot2[k] = partsPlot2[k]/partsCodedNumber2[k]
    print(partsPlot1.shape)
    #gr.images(partsPlot1[0:200],vmin = 0,vmax = 1)
    #gr.images(partsPlot2,vmin = 0,vmax = 1)
    print(partsCodedNumber1)
    print("-----------------")
    print(partsCodedNumber2)
    return partsPlot1,partsPlot2,extractedFeature
def investigate():
    partsPlot1,partsPlot2,extractedFeature = displayParts()
    import matplotlib.pylab as plot
    for partIndex in range(20):
        test = []
        smallerPart = []
        for k in range(1000):
            x = extractedFeature[4][k]
            for m in range(8):
                for n in range(8):
                    if(x[m,n,0] == partIndex):
                        test.append((k,m,n))
                        smallerPart.append(extractedFeature[2][k,3 * m + 1,3 * n + 1]) 
        number = np.zeros(200)
        for x in smallerPart:
            if(x!=-1):
                number[x]+=1
        #plot1 = plot.figure(partIndex)
        #plot.plot(number)
        #plot.savefig('frequency %i.png' %partIndex)
        #plot.close()
        index = np.where(number > 100)[0]
        partNew2 = np.ones((index.shape[0] + 1,6,6))
        partNew2[0] = partsPlot2[partIndex]
        for i in range(index.shape[0]):
            partNew2[i + 1,0:4,0:4] = partsPlot1[index[i],:,:]
        fileString = 'part%i.png' %partIndex
        gr.images(partNew2,zero_to_one=False, show=False,vmin = 0, vmax = 1, fileName = fileString) 
      
      
      
      
         
