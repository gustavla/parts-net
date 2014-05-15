from __future__ import division, print_function,absolute_import
import pylab as plt
import amitgroup.plot as gr
import numpy as np
import amitgroup as ag
import os
import pnet
import matplotlib.pylab as plot
def extract(ims,allLayers):
    #print(allLayers)
    curX = ims
    for layer in allLayers:
        #print('-------------')
        #print(layer)
        curX = layer.extract(curX)
        #print(np.array(curX).shape)
        #print('------------------')
    return curX

def partsPool(originalPartsRegion, numParts):
    partsGrid = np.zeros((1,1,numParts))
    for i in range(originalPartsRegion.shape[0]):
        for j in range(originalPartsRegion.shape[1]):
            if(originalPartsRegion[i,j]!=-1):
                partsGrid[0,0,originalPartsRegion[i,j]] = 1
    return partsGrid



def test(ims,labels,net):
    yhat = net.classify((ims,3000))
    return yhat == labels
    


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
    
    imgRegion= [[] for x in range(numParts)]
    partsRegion = [[] for x in range(numParts)]

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
    
        
    ##second-layer parts
    numSecondLayerParts = 30
    allPartsLayer = [[pnet.PartsLayer(numSecondLayerParts,(1,1),
                        settings=dict(outer_frame = 0, 
                        threshold = 5, 
                        sample_per_image = 1, 
                        max_samples=10000, 
                        min_prob = 0.005))] 
                        for i in range(numParts)]
    allPartsLayerImg = np.zeros((numParts,numSecondLayerParts,12,12))
    allPartsLayerImgNumber = np.zeros((numParts,numSecondLayerParts))
    
    zeroParts = 0
    
    for i in range(numParts):
        print("test")
        allPartsLayer[i][0].train_from_samples(np.array(partsRegion[i]),None)
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



    """
    Visualize the SuperParts
    """
    settings = {'interpolation':'nearest','cmap':plot.cm.gray,}
    settings['vmin'] = 0
    settings['vmax'] = 1
    plotData = np.ones((14*100+2,14*(numSecondLayerParts + 1)+2))*0.8
    visualShiftParts = 0
    if 0:
        allPartsPlot = np.zeros((20,numSecondLayerParts + 1,12,12))
        gr.images(partsPlot.reshape(numParts,6,6),zero_to_one=False,vmin = 0, vmax = 1)
        allPartsPlot[:,0] = 0.5
        allPartsPlot[:,0,3:9,3:9] = partsPlot[20:40]
        allPartsPlot[:,1:,:,:] = allPartsLayerImg[20:40]
        gr.images(allPartsPlot.reshape(20 * (numSecondLayerParts + 1),12,12),zero_to_one=False, vmin = 0, vmax =1)
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



    """
    Train A Class-Model Layer
    """
    
    digits = range(10)
    sup_ims = []
    sup_labels = []
    
    classificationTrainingNum = 100
    for d in digits:
        ims0 = ag.io.load_mnist('training', [d], selection = slice(classificationTrainingNum), return_labels = False)
        sup_ims.append(ims0)
        sup_labels.append(d * np.ones(len(ims0),dtype = np.int64))
    sup_ims = np.concatenate(sup_ims, axis = 0)
    sup_labels = np.concatenate(sup_labels,axis = 0)
    

    curX = extract(sup_ims,allLayer[0:2])[0]
    #print(curX.shape)
    curX = curX.reshape(curX.shape[0:3])
    secondLevelCurx = np.zeros((10 * classificationTrainingNum,17,17,1,1,numParts))
    secondLevelCurxCenter = np.zeros((10 * classificationTrainingNum,17,17))
    for i in range(10 * classificationTrainingNum):
        codeParts = curX[i]
        for m in range(23)[3:20]:
            for n in range(23)[3:20]:
                secondLevelCurx[i,m-3,n-3] = partsPool(codeParts[m-3:m+4,n-3:n+4],numParts)
                secondLevelCurxCenter[i,m-3,n-3] = codeParts[m,n]

    thirdLevelCurx = np.zeros((10 * classificationTrainingNum, 17,17))
    for i in range(int(10 * classificationTrainingNum)):
        for m in range(17):
            for n in range(17):
                if(secondLevelCurxCenter[i,m,n]!=-1):
                    firstLevelPartIndex = secondLevelCurxCenter[i,m,n]
                    #print(firstLevelPartIndex)
                    firstLevelPartIndex = int(firstLevelPartIndex)
                    extractedFeaturePart = extract(np.array(secondLevelCurx[i,m,n][np.newaxis,:],dtype = np.uint8),allPartsLayer[firstLevelPartIndex])[0]
                    #print("secondLayerExtraction")
                    #print(extractedFeaturePart.shape)
                    thirdLevelCurx[i,m,n] = int(numSecondLayerParts * firstLevelPartIndex + extractedFeaturePart)
                    #print(numSecondLayerParts,firstLevelPartIndex,extractedFeaturePart,thirdLevelCurx[i,m,n])
                else:
                    thirdLevelCurx[i,m,n] = -1
    
    print(thirdLevelCurx.shape)
    #return thirdLevelCurx,allPartsLayerImg 
    if 1:
        classificationLayers = [
                            pnet.PoolingLayer(shape = (4,4),strides = (4,4)),
                            pnet.MixtureClassificationLayer(n_components = 1, min_prob = 1e-7, block_size = 200)
                            #pnet.SVMClassificationLayer(C=1.0)
        ]
        classificationNet = pnet.PartsNet(classificationLayers)
        classificationNet.train((np.array(thirdLevelCurx[:,:,:,np.newaxis],dtype = np.int64),int(numParts * numSecondLayerParts)),sup_labels[:])
        print("Training Success!!")    
    
    if 1:    
        testImg,testLabels = ag.io.load_mnist('testing')
        testingNum = testLabels.shape[0]
        
        curTestX = extract(testImg, allLayer[0:2])[0]
        curTestX = curTestX.reshape(curTestX.shape[0:3])
        secondLevelCurTestX = np.zeros((testingNum, 17,17,1,1,numParts))
        secondLevelCurTestXCenter = np.zeros((testingNum, 17,17))
        
        import time
        start = time.time()
        for i in range(testingNum):
            codeParts = curTestX[i]
            for m in range(23)[3:20]:
                for n in range(23)[3:20]:
                    secondLevelCurTestX[i,m-3,n-3] = partsPool(codeParts[m-3:m+4,n-3:n+4],numParts)
                    secondLevelCurTestXCenter[i,m-3,n-3] = codeParts[m,n]
        afterPool = time.time()
        print(afterPool - start)
        thirdLevelCurTestX = np.zeros((testingNum, 17, 17))
        for i in range(testingNum):
            for m in range(17):
                for n in range(17):
                    if(secondLevelCurTestXCenter[i,m,n]!=-1):
                        firstLevelPartIndex = int(secondLevelCurTestXCenter[i,m,n])
                        extractedFeaturePart = extract(np.array(secondLevelCurTestX[i,m,n][np.newaxis,:],dtype = np.uint8),allPartsLayer[firstLevelPartIndex])[0]
                        thirdLevelCurTestX[i,m,n] = numSecondLayerParts * firstLevelPartIndex + extractedFeaturePart
                    else:
                        thirdLevelCurTestX[i,m,n] = -1
        end = time.time()
        print(end-afterPool)
        print(thirdLevelCurTestX.shape)
        testImg_Input = np.array(thirdLevelCurTestX[:,:,:,np.newaxis],dtype = np.int64) 
        testImg_batches = np.array_split(testImg_Input,200)
        testLabels_batches = np.array_split(testLabels, 200)
        
        args = [tup + (classificationNet,) for tup in zip(testImg_batches,testLabels_batches)]
        
        corrects = 0
        total = 0
        
        def format_error_rate(pr):
            return "{:.2f}%".format(100 * (1-pr))

        print("Testing Starting...")
        for i, res in enumerate(pnet.parallel.starmap_unordered(test,args)):
            if i !=0 and i % 20 ==0:
                print("{0:05}/{1:05} Error rate: {2}".format(total, len(ims),format_error_rate(pr)))

            corrects += res.sum()
            total += res.size

            pr = corrects / total
        
        print("Final error rate:", format_error_rate(pr))













