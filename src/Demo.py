import math
import numpy as np
import cv2
from scipy.signal import argrelextrema
import glob
import os
from Model import Model, DecoderType
from DataLoader import DataLoader, Batch
#import matplotlib.pyplot as plt
from SamplePreprocessor import preprocess
import csv
from pyzbar.pyzbar import decode

inputfolder='..//Inputs'
outputfolder='..//Outputs'

startsrow=3;
nrow=14;



class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = '../data/test2.png'
	fnCorpus = '../data/corpus.txt'




decoderType = DecoderType.BestPath
model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
		


names = [os.path.basename(x) for x in glob.glob(inputfolder+'/*.jpg')]

for fno in range(0,len(names)):
    
    print("Processing for "+names[fno])
    
    img = cv2.imread(inputfolder+'/'+names[fno])

    # rorate image if its row is greater than column
    if img.shape[0]>img.shape[1]:
        img=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    barcode=decode(gray)
    if len(barcode)>0:
        i1=barcode[0][2].top
        i2=barcode[0][2].top+barcode[0][2].height
        j1=barcode[0][2].left
        j2=barcode[0][2].left+barcode[0][2].width
        img_code=gray[i1-10:i2+10,j1-10:j2+10].copy()
        gray[i1-10:i2+10,j1-10:j2+10]=255
    else:
        img_code=np.zeros((50,50))
        tmp=[]
        tmp.append("No Barcode")
        barcode.append(tmp)


    lsd = cv2.createLineSegmentDetector(0)
    dlines = lsd.detect(gray)

    filter_min_dist=100
    filter_min_tng=30
    window_len=20

    acc=np.zeros((gray.shape[0],))
    suma=0
    sumw=0
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))   

        if x1<x0:
            tmp=x1
            x1=x0
            x0=tmp
            tmp=y1
            y1=y0
            y0=tmp
    
        a = (x0-x1) * (x0-x1)
        b = (y0-y1) * (y0-y1)
        c=a+b
        dst=math.sqrt(c)
        # filter for detecting almost horizontal lines
        if dst>filter_min_dist and a>filter_min_tng*b :
            suma+= np.arctan2(y1-y0,x1-x0)*dst
            sumw+=dst 

    ang=180*(suma/sumw)/np.pi
    center=(gray.shape[1]/2,gray.shape[0]/2)
    H=cv2.getRotationMatrix2D(center, ang, 1)
    img=cv2.warpAffine(img,H,(gray.shape[1],gray.shape[0]))

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)
    dlines = lsd.detect(gray)


    filter_min_dist=30
    filter_min_tng=100
    window_len=20

    acc=np.zeros((gray.shape[0],))

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3])) 

    
        a = (x0-x1) * (x0-x1)
        b = (y0-y1) * (y0-y1)
        c=a+b
        # filter for detecting almost horizontal lines
        if math.sqrt(c)>filter_min_dist and a>filter_min_tng*b :  
            # calculate accumultor for horizontal lines row number     
            incr=(max(x0,x1)-min(x0,x1))/(1+(max(y0,y1)-min(y0,y1)))
            for i in range(min(y0,y1),max(y0,y1)+1):
                acc[i]+=incr

    # smoothed the acculmulator
    s=np.r_[np.zeros((int(window_len/2),)),acc,np.zeros((int(window_len/2),))]
    w=np.hamming(window_len)
    y=np.convolve(w/w.sum(),s,mode='valid')

    # find local maxima
    linerows=argrelextrema(y, np.greater)

    # eliminate local maxima points if they are less then half of maximum
    activelines=[]
    for i in range(0,len(linerows[0])):
        if y[linerows[0][i]]>y.max()/3:
             activelines.append(i)


    # do it for vertical lines too
    accV=np.zeros((gray.shape[1],))
    filter_min_tng=50
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3])) 

    
        a = (x0-x1) * (x0-x1)
        b = (y0-y1) * (y0-y1)
        c=a+b
        # filter for detecting almost vertical lines
        if math.sqrt(c)>filter_min_dist and b>filter_min_tng*a :  
            # calculate accumultor for vertical lines row number     
            incr=(max(y0,y1)-min(y0,y1))/(1+(max(x0,x1)-min(x0,x1)))
            for i in range(min(x0,x1),max(x0,x1)+1):
                accV[i]+=incr

    # smoothed the acculmulator
    s=np.r_[np.zeros((int(window_len/2),)),accV,np.zeros((int(window_len/2),))]
    w=np.hamming(window_len)
    y=np.convolve(w/w.sum(),s,mode='valid')

    #plt.plot(y)
    #plt.show()

    # find local maxima
    linerowsV=argrelextrema(y, np.greater)

    # eliminate local maxima points if they are less then half of maximum
    activelinesV=[]
    for i in range(0,len(linerowsV[0])):
        if y[linerowsV[0][i]]>y.max()/3:
             activelinesV.append(i)



    # crop image
    directory=outputfolder +'/' + names[fno][0:-4]
    if not os.path.exists(directory):
        os.makedirs(directory)

    csvfile=open(outputfolder +'/' + names[fno][0:-4] +'.csv', mode='w')
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow(["Barcode:",str(barcode[0][0])])
    fname=outputfolder +'/' + names[fno][0:-4]+ "/barcode.jpg"
    cv2.imwrite(fname,img_code)

    for i in range(startsrow,min(startsrow+nrow,len(activelines))):
        rowtext=[]        
        for j in range(1,len(activelinesV)):
            tmp=gray[linerows[0][activelines[i-1]]:linerows[0][activelines[i]],linerowsV[0][activelinesV[j-1]]:linerowsV[0][activelinesV[j]]]
            tmp=tmp[:,2:-2]
            fname=outputfolder +'/' + names[fno][0:-4]+ "/"+str(i)+"_row_"+str(j)+"_col.jpg"
            cv2.imwrite(fname,tmp)
            img = preprocess(tmp, Model.imgSize)
            batch = Batch(None, [img] * Model.batchSize) 
            recognized = model.inferBatch(batch) 
            rowtext.append(recognized[0])
            #print('Recognized:', '"' + recognized[0] + '"')  
        csv_writer.writerow(rowtext) 
    csvfile.close()
    #plt.plot(y)
    #plt.show()
  