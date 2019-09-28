import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def extraction(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    color_img=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ##cv2.imshow('color',color_img)
    ##cv2.imshow('gray',gray)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #kernel = np.ones((5,5),np.uint8)
    eroded=cv2.medianBlur(img,5)
    #cv2.imshow('median_Blurred',eroded)
    hsv=cv2.cvtColor(eroded,cv2.COLOR_BGR2HSV)
    lb=np.array([00,0,0])
    ub=np.array([170,255,255])
    lb1=np.array([0,180,20])
    ub1=np.array([255,255,255])
    mask=cv2.inRange(eroded,lb,ub)
    res=cv2.bitwise_and(eroded,eroded,mask=mask)

    mask2=cv2.inRange(res,lb1,ub1)
    ##cv2.imshow('msk2',mask2)
    cv2.imshow('msk',mask)
    cv2.imshow('msk2',mask2)
    res2=cv2.bitwise_and(res,res,mask=mask2)
    res2=res2[50:(len(res2)-50),:]
    return res2

def bounding_box(img):
##img=cv2.imread('cropped0.jpg')

    ##res2=res2[50:(len(res2)-30),:]
    print("yo")
    cv2.imshow('omh',img)
    res2=extraction(img)
    l=np.zeros(((len(res2[0])-1),), dtype=int)
    for j in range(0,(len(res2[0])-1)):
        for i in range(0,(len(res2)-1)):
            if((int(res2[i][j][0])+int(res2[i][j][1])+int(res2[i][j][2]))>0):
                l[j]=l[j]+1;
    maximum=np.amax(l)
    print(maximum)
    result = np.where(l == np.amax(l))
    b=0
    e=0
    x=result[0][0]

    midx=len(result[0])/2
    mid=result[0][math.floor(midx)]
    ini=0
    row_no=[]
    x0=(len(res2[0])-1)
    x1=(len(res2[0])-1)
    res2_hsv=cv2.cvtColor(res2,cv2.COLOR_BGR2HSV)
    cv2.imshow('res2_hsv',res2_hsv)
    ##for i in range((len(res2_hsv)-1),0):
    i=(len(res2_hsv)-1)
    while i > 0:
        #print(i)
        sum=np.sum(res2_hsv[i][mid])
        if(int(abs(sum-ini))>int((0.3*ini))):
            #print('YO')
            ini=sum
            x0=x1
            x1=i+1
            row_no.append(x0)
            row_no.append(x1)
        i=i-1
    print(row_no)
    
    diff=abs(row_no[3]-row_no[2])
    x1=row_no[3]
    x0=row_no[2]
    print('diff %d',diff)
    i=4
    while i < (len(row_no)-1):
        if((abs(row_no[i+1] - row_no[i-1]))>diff):
            diff=abs(row_no[i+1]-row_no[i])
            x1=row_no[i+1]
            x0=row_no[i]
            print('x0 %d and x1 %d',x0,x1)
            print('diff2',diff)
        i=i+2


    mid2=abs((x0-x1)/2)

    while x > 0  and l[x]>=0.60*maximum :
        b=x
        x=x-1;
    for x in range (result[0][len(result[0])-1],(len(res2)-1)):
        if(l[x]>=0.60*maximum):
            e=x
        else:
            break;
    w = e if e < b else b       
    print(result,b,e,w)
    ext1=abs((x1)*0.2)
    ext1D=int(x1-ext1)-30
    ext2=abs((len(res2)-x0)*0.2)
    ext2D=int(x0+ext2)

    print('ext2D %d',ext2D)
    print('ext1D %d',ext1D)
    cv2.rectangle(res2,((result[0][math.floor(midx)]-80),ext1D),((result[0][math.floor(midx)]+80),(ext2D)),(0,0,255),1)
    #cv2.imshow('res2.jpg',res2)
    final_img=res2[(ext1D+15):(ext2D-10),(result[0][math.floor(midx)]-65):(result[0][math.floor(midx)]+65)]
    ##Bounding box size matrix
    height=[(ext1D)-30,(ext2D)-13]
    width= [(result[0][math.floor(midx)]-80),(result[0][math.floor(midx)]+80)]
    
    #cv2.imshow('final',final_img)
    return height,width

##    ret,thresh = cv2.threshold(res2,25,255,cv2.THRESH_BINARY)
##    ##cv2.imshow('thresh',thresh)
##    cv2.imshow('image',res2)
##    ##cv2.imshow('msk',mask)
    cv2.waitKey(0)
