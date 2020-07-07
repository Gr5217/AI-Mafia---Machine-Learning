import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier('frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('Nose18x15.xml')

img = cv2.imread('Before.png')
glasses=cv2.imread("glasses.png",-1)
mustache=cv2.imread("mustache.png",-1)
print('Original Dimension : ',img.shape)
faces=face_cascade.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5)
print(faces)

img=cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
for (x,y,w,h) in faces:
    roi_color = img[y:y+h, x:x+w]
    eyes = eyes_cascade.detectMultiScale(roi_color)
    print(eyes)
    nose = nose_cascade.detectMultiScale(roi_color)
    print(nose)
    sense=[eyes,nose]
    print(sense)
    ex,ey,ew,eh,nx,ny,nw,nh =  eyes[0][0],eyes[0][1],eyes[0][2],eyes[0][3],nose[0][0],nose[0][1],nose[0][2],nose[0][3]
    roi_eyes = roi_color[ey:ey + eh, ex:ex + ew]
    dim1=(eh,ew+ex)
    roi_nose = roi_color[ny:ny + nh, nx:nx + nw]
    dim2 = (nh+20, nw)
    glasses=cv2.resize(glasses,dim1)
    mustache=cv2.resize(mustache,dim2)

    gw,gh,gc=glasses.shape
    mw, mh, mc = mustache.shape
    for i in range(0,gw):
        for j in range(0,gh):
            if glasses[i,j][3]!=0:
                roi_color[ey+i,ex+j]=glasses[i,j]
    for i in range(0, mw):
        for j in range(0, mh):
            if mustache[i,j][3]!=0:
                roi_color[ny+int(nh/2.0)+i,nx+j]=mustache[i,j]
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()