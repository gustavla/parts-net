from __future__ import division, print_function,absolute_import
import pylab as plt
import amitgroup.plot as gr
import numpy as np
import amitgroup as ag
import os
import pnet
import matplotlib.pylab as plot
def extract(ims,allLayers):
    print(allLayers)
    curX = ims
    for layer in allLayers:
        print('-------------')
        print(layer)
        curX = layer.extract(curX)
        print(np.array(curX).shape)
        print('------------------')
    return curX

def test2():
    #TODO: NOT RIGHT#
    X = np.load('testNew.npy')
    model = X.item()
    net = pnet.PartsNet.load_from_dict(model)
    allLayer = net.layers
    print(allLayer)
    ims, labels = ag.io.load_mnist('training')
    extractedParts = extract(ims[0:1000],allLayer[0:2])
    #return extractedParts
    allParts = extractedParts[0]
    parts_layer = allLayer[1]
    parts = parts_layer._parts.reshape(100,6,6)
    #for i in range(200):
    ims = ims[0:1000]
    labels = labels[0:1000]
    #print(ims.shape)
    classifiedLabel = net.classify(ims)
    #print out all the misclassified  images

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
    # load the trained Image
    #X =  np.load('testNew.npy')
    X = np.load('testNewPooling44.npy')
    model = X.item()
    # get the parts Layer
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
    partsPlot1 = np.zeros((numParts1,6,6))
    partsCodedNumber1 = np.zeros(numParts1)
    partsPlot2 = np.zeros((numParts2,12,12))
    partsCodedNumber2 = np.zeros(numParts2)

    for i in range(1000):
        codeParts1 = extractedParts1[i].reshape(extractedParts1[i].shape[0:2])
        codeParts2 = extractedParts2[i].reshape(extractedParts2[i].shape[0:2])
        for m in range(23):
            for n in range(23):
                if(codeParts1[m,n]!=-1):
                    partsPlot1[codeParts1[m,n]]+=ims[i,m:m+6,n:n+6] 
                    partsCodedNumber1[codeParts1[m,n]]+=1
        for p in range(23)[3:19]:
            for q in range(23)[3:19]:
                if(codeParts2[p,q]!=-1):
                    partsPlot2[codeParts2[p,q]]+=ims[i,p - 3:p + 9,q - 3: q + 9]
                    partsCodedNumber2[codeParts2[p,q]]+=1
                #if(codeParts2[p,q,1]!=-1):
                #    partsPlot2[codeParts2[p,q,1]]+=ims[i,p:p+10,q:q+10]
                #    partsCodedNumber2[codeParts2[p,q,1]]+=1
    for j in range(numParts1):
        partsPlot1[j] = partsPlot1[j]/partsCodedNumber1[j]
    for k in range(numParts2):
        partsPlot2[k] = partsPlot2[k]/partsCodedNumber2[k]
    print(partsPlot1.shape)
    gr.images(partsPlot1,vmin = 0,vmax = 1)
    gr.images(partsPlot2[0:1000],vmin = 0,vmax = 1)
    print(partsCodedNumber1)
    print("-----------------")
    print(partsCodedNumber2)
    return partsPlot1,partsPlot2,extractedFeature
def investigate():
    partsPlot1,partsPlot2,extractedFeature = displayParts()
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

def partsPool(originalPartsRegion, numParts):
    partsGrid = np.zeros((1,1,numParts))
    for i in range(originalPartsRegion.shape[0]):
        for j in range(originalPartsRegion.shape[1]):
            if(originalPartsRegion[i,j]!=-1):
                partsGrid[0,0,originalPartsRegion[i,j]] = 1
    return partsGrid

def trainPOP():
    X = np.load("test4.npy") 
    model = X.item()
    # get num of Parts
    numParts = model['layers'][1]['num_parts'] 
    net = pnet.PartsNet.load_from_dict(model)
    allLayer = net.layers
    ims,labels = ag.io.load_mnist('training')
    trainingDataNum = 1000
    extractedFeature = extract(ims[0:trainingDataNum],allLayer[0:2])[0]
    print(extractedFeature.shape)
    extractedFeature = extractedFeature.reshape(extractedFeature.shape[0:3])
    partsPlot = np.zeros((numParts,6,6))
    partsCodedNumber = np.zeros(numParts)
        
    #every list corresponding to the larger region surrounding 10x10 region of the 5*5 region coded by this part 
    imgRegion = [[] for x in range(numParts)]
    partsRegion = [[] for x in range(numParts)]    
    #Part Visualize#
    for i in range(trainingDataNum):
        codeParts = extractedFeature[i]
        for m in range(23):
            for n in range(23):
                if(codeParts[m,n]!=-1):
                    partsPlot[codeParts[m,n]]+=ims[i,m:m+6,n:n+6]
                    partsCodedNumber[codeParts[m,n]]+=1    
    for j in range(numParts):
        partsPlot[j] = partsPlot[j]/partsCodedNumber[j]


    secondLayerCodedNumber = 0
    if 1:
        for i in range(trainingDataNum):
            codeParts = extractedFeature[i]
            for m in range(23)[3:20]:
                for n in range(23)[3:20]:
                    if(codeParts[m,n]!=-1):
                        imgRegion[codeParts[m,n]].append(ims[i,m-3:m+9,n-3:n+9])
                        secondLayerCodedNumber+=1
                        partsGrid = partsPool(codeParts[m-3:m+4,n-3:n+4],numParts)
                        partsRegion[codeParts[m,n]].append(partsGrid)

    for i in range(numParts):
        print(len(partsRegion[i]))
   
    ##Second-Layer Parts
    numSecondLayerParts = 20  
    allPartsLayer = [[pnet.PartsLayer(numSecondLayerParts,(1,1),settings=dict(outer_frame=0,threshold=5,
                                                            sample_per_image=1,
                                                            max_samples=10000,
                                                            min_prob=0.005))] for i in range(numParts)]
    allPartsLayerImg = np.zeros((numParts,numSecondLayerParts,12,12)) 
    
    allPartsLayerImgNumber = np.zeros((numParts,numSecondLayerParts))
   
    print("====================================================") 
    
    zeroParts = 0
    for i in range(numParts):
        print("test")
        allPartsLayer[i][0].train_from_samples(np.array(partsRegion[i]),None)
        print(np.array(partsRegion[i]).shape)
        extractedFeaturePart = extract(np.array(partsRegion[i],dtype = np.uint8),allPartsLayer[i])[0]
        print(extractedFeaturePart.shape)
        for j in range(len(partsRegion[i])):
            if(extractedFeaturePart[j,0,0,0]!=-1):
                partIndex = extractedFeaturePart[j,0,0,0]
                allPartsLayerImg[i,partIndex]+=imgRegion[i][j]
                allPartsLayerImgNumber[i,partIndex]+=1
            else:
                zeroParts+=1
    for i in range(numParts):
        for j in range(numSecondLayerParts):
            allPartsLayerImg[i,j] = allPartsLayerImg[i,j]/allPartsLayerImgNumber[i,j] 
        print(allPartsLayer[i][0]._weights)
    #print(zeroParts)
    #print(np.sum(allPartsLayerImgNumber),secondLayerCodedNumber)
    settings = {'interpolation':'nearest','cmap':plot.cm.gray,}
    settings['vmin'] = 0
    settings['vmax'] = 1
    plotData = np.ones((14*100+2,14*(numSecondLayerParts + 1)+2))*0.8
    visualShiftParts = 0
    if 0:
        allPartsPlot = np.zeros((20,11,12,12))
        gr.images(partsPlot.reshape(numParts,6,6),zero_to_one=False,vmin = 0, vmax = 1)
        allPartsPlot[:,0] = 0.5
        allPartsPlot[:,0,3:9,3:9] = partsPlot[20:40]
        allPartsPlot[:,1:,:,:] = allPartsLayerImg[20:40]
        gr.images(allPartsPlot.reshape(220,12,12),zero_to_one=False, vmin = 0, vmax =1)
    elif 0:
        for i in range(numSecondLayerParts + 1):
            for j in range(100):
                if i == 0:
                    plotData[5 + j * 14:11 + j * 14, 5 + i * 14: 11 + i * 14] = partsPlot[j+visualShiftParts]
                else:
                    plotData[2 + j * 14:14 + j * 14,2 + i * 14: 14 + i * 14] = allPartsLayerImg[j+visualShiftParts,i-1]
        plot.figure(figsize=(10,40))
        plot.axis('off')
        plot.imshow(plotData, **settings)
        plot.savefig('test.pdf',format='pdf',dpi=900)
    else:
        pass









     
      
      
         
