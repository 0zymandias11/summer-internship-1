import cv2
import te2
import find_angle
import math
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
##for i in range (0,1524):
##img=cv2.imread('cropped'+str(0)+'.jpg')
####if i==0:
##global h,w
##h,w=te2.bounding_box(img)
img=cv2.imread('cropped'+str(4)+'.jpg')
cv2.imshow('omhhhhhhh',img)
h1,w1=te2.bounding_box(img)
midw=0
midh=0
width_mid_body_x=0
checker=0
k=0
waist_angle=[]
waist_coordinate=[]
knee_coordinate1=[]
knee_coordinate2=[]
for iteration in range (4,300):
    h=h1.copy()
    w=w1.copy()
    w[1]=w[1]-50
    print(iteration)
    img2=cv2.imread('cropped'+str(iteration)+'.jpg')
    img2 = cv2.imread('cropped'+str(iteration)+'.jpg')
    extracted=te2.extraction(img2)
    cv2.imshow('extracted',extracted)
    cv2.imshow('roisss',extracted[h[0]:h[1],w[0]:w[1]+25])
    hsv=cv2.cvtColor(extracted,cv2.COLOR_BGR2HSV)
    k=0
    
    for i in range(h[0],h[1]-20):
        sum=0
        for j in range((w[1]-5),w[1]):
            sum+=(int(hsv[i][j][0]))+(int(hsv[i][j][1]))+(int(hsv[i][j][2]))
        if(sum>=100):
           k=-1
           w[1]=w[1]+10
           break
        if(k== -1):
            break 
    k=0
    for i in range(h[0],h[1]-20):
        sum=0
        for j in range(w[0],(w[0]+30)):
            sum+=(int(hsv[i][j][0]))+(int(hsv[i][j][1]))+(int(hsv[i][j][2]))
        if(sum>=100):
           k=-1
           w[0]=w[0]-20
           break
        if(k== -1):
            break

## boundary data has been extracted upto this point

##Skeletal work starts
    width_mid_body=[]
    wl=0
    ##if iteration==0 :
    midw=int((w[1]-w[0])/2)
    midh=int((h[1]-h[0])/2)
    for l in range(int(w[0]),int(w[1])):
        #print (l)
        if((int(hsv[midh][l][0]))+(int(hsv[midh][l][1]))+(int(hsv[midh][l][2]))>100):
            wl+=1
            width_mid_body.append(l)
    ##print('jlkkl ',len(width_mid_body),' jhjkh ',w[0],' kjnk ',w[1],' midh ',midh)
    width_mid_body_x=width_mid_body[int(wl/2)-1]


    
        

    temp=0
    
    for i in range(h[0],int(h[1]*0.6)):
        sum=0
        for j in range(w[0],w[1]):
            if((int(hsv[i][j][0]))+(int(hsv[i][j][1]))+(int(hsv[i][j][2]))>100):
                sum+=1
        if(sum>=10):
            k=-1
            temp=i
            break
    #head and waist
    body_height=h[1]-i
    head_centroid_height=h[1]-(body_height-(int(0.065*body_height)))
    waist_centroid_height=head_centroid_height+int(0.430*body_height)
    cv2.circle(extracted,(width_mid_body_x,head_centroid_height),2,(255,0,0),-1)
    cv2.circle(extracted,(width_mid_body_x,waist_centroid_height),2,(255,0,0),-1)

    cv2.line(extracted,(width_mid_body_x,head_centroid_height),(width_mid_body_x,waist_centroid_height),(0,0,255),1)
    #knee (this one is a fuckin approximation and a bit erroneous) if u have a better way to find the knee centroid plss do tell me

    knee_centroid_height=waist_centroid_height+int(0.190*body_height)
        #for forward knee
    wl=0
    w2=0
    fknee_width=[]
    for l in range(width_mid_body_x,w[1]):
        if((int(hsv[knee_centroid_height][l][0]))+(int(hsv[knee_centroid_height][l][1]))+(int(hsv[knee_centroid_height][l][2]))>100):
            wl+=1
            fknee_width.append(l)
        #if the knee moves backwards
    if wl<=15:
        wl=0
        fknee_width.clear()
        for l in range (w[0],width_mid_body_x):
            if((int(hsv[knee_centroid_height][l][0]))+(int(hsv[knee_centroid_height][l][1]))+(int(hsv[knee_centroid_height][l][2]))>100):
                wl+=1
                fknee_width.append(l)
    
    fknee_mid_x=fknee_width[int(wl/2)-1]
    print
    cv2.circle(extracted,(fknee_mid_x,knee_centroid_height),2,(255,0,0),-1)
        #for back knee
    wl=0
    bknee_width=[]
    
    for l in range(w[0],width_mid_body_x):
        if((int(hsv[knee_centroid_height][l][0]))+(int(hsv[knee_centroid_height][l][1]))+(int(hsv[knee_centroid_height][l][2]))>100):
            wl+=1
            bknee_width.append(l)
    if wl <=10:
        wl=0
        bknee_width.clear()
        for l in range(width_mid_body_x,w[1]):
            if((int(hsv[knee_centroid_height][l][0]))+(int(hsv[knee_centroid_height][l][1]))+(int(hsv[knee_centroid_height][l][2]))>100):
                wl+=1
                bknee_width.append(l)
    bknee_mid_x=bknee_width[int(wl/2)-1]


    cv2.circle(extracted,(bknee_mid_x,knee_centroid_height),2,(255,0,0),-1)
    cv2.line(extracted,(width_mid_body_x,waist_centroid_height),(bknee_mid_x,knee_centroid_height),(0,0,255),1)
    cv2.line(extracted,(width_mid_body_x,waist_centroid_height),(fknee_mid_x,knee_centroid_height),(0,0,255),1)
    waist_coordinate.append((width_mid_body_x,waist_centroid_height))
    knee_coordinate1.append((bknee_mid_x,knee_centroid_height))
    knee_coordinate2.append((fknee_mid_x,knee_centroid_height))
        ##print('f_knee ', fknee_mid_x, ' bknee ',bknee_mid_x)
        

    #for ankle
    ankle_centroid_height=waist_centroid_height+int(0.431*body_height)
    wl=0
    fankle_width=[]
    for l in range(width_mid_body_x,w[1]):
        if((int(hsv[ankle_centroid_height][l][0]))+(int(hsv[ankle_centroid_height][l][1]))+(int(hsv[ankle_centroid_height][l][2]))>100):
            wl+=1
            fankle_width.append(l)
    if wl==0:
        fankle_mid_x=width_mid_body_x
    else:
        fankle_mid_x=fankle_width[int(wl/2)-1]
    print
    cv2.circle(extracted,(fankle_mid_x,ankle_centroid_height),2,(255,0,0),-1)
        #for back ankle
    wl=0
    bankle_width=[]
    
    for l in range(w[0],bknee_mid_x):
        if((int(hsv[ankle_centroid_height][l][0]))+(int(hsv[ankle_centroid_height][l][1]))+(int(hsv[ankle_centroid_height][l][2]))>100):
            wl+=1
            bankle_width.append(l)
    if wl==0:
        bankle_mid_x=bknee_mid_x-5
    else:
        bankle_mid_x=bankle_width[int(wl/2)-1]
        
    cv2.circle(extracted,(bankle_mid_x,ankle_centroid_height),2,(255,0,0),-1)
    cv2.line(extracted,(fknee_mid_x,knee_centroid_height),(fankle_mid_x,ankle_centroid_height),(0,0,255),1)
    cv2.line(extracted,(bknee_mid_x,knee_centroid_height),(bankle_mid_x,ankle_centroid_height),(0,0,255),1)

    cv2.rectangle(extracted,(w[0],h[0]),(w[1],h[1]),(0,0,255),1)
    
    if(iteration==200):
        break
##boundary drawing
    
    cv2.imshow('frame',extracted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
waist_angle,knee_coordinate1=find_angle.angle_calculation (waist_coordinate,knee_coordinate1,knee_coordinate2)
filtered=medfilt(waist_angle,kernel_size=3)
x=np.linspace(1,(len(knee_coordinate1)),num=len(knee_coordinate1))

f = interp1d(x,filtered,kind='cubic')
xnew=np.linspace(1,(len(knee_coordinate1)),num=(len(knee_coordinate1)))
y=f(xnew)
f2=medfilt(y,kernel_size=13)
plt.plot(xnew,f2)
plt.xlabel("Number of frames ---->")
plt.ylabel("Hip angle (in Degrees) ---->")
plt.title("Hip angle from Depth video")
#cap.release()
#cv2.destroyAllWindows()
##import csv
##myFile = open('subject3_depth.csv', 'a')
##with myFile:
##    writer = csv.writer(myFile)
##    writer.writerows(map(lambda x: [x], waist_angle[0:30]))
