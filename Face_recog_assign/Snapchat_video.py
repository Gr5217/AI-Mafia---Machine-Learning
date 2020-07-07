import cv2
import numpy as np

from utils import CFEVideoConf, image_resize

cap =cv2.VideoCapture(0)

save_path='saved-media/Snapchat_video.mp4'
frames_per_seconds=24
config=CFEVideoConf(cap, filepath=save_path,res='720p')
out=cv2.VideoWriter(save_path,config.video_type,frames_per_seconds,config.dims)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyes_cascade=cv2.CascadeClassifier('frontalEyes35x16.xml')
nose_cascade=cv2.CascadeClassifier('Nose18x15.xml')
glasses=cv2.imread('glasses.png',-1)
mustache=cv2.imread('mustache.png',-1)

while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+h]
        roi_color = frame[y:y + h, x:x + h]
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)

        eyes=eyes_cascade.detectMultiScale(roi_gray,scaleFactor=1.5,minNeighbors=5)
        for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ey + eh, ex + ew), (127, 255, 0), 3)
            roi_eyes=roi_gray[ey:ey+eh,ex:ex+ew]
            glasses1=image_resize(glasses.copy(), width=eh)

            gw,gh,gc=glasses1.shape
            for i in range(0,gw):
                for j in range(0,gh):
                    #print(glasses[i,j])
                    if glasses1[i,j][3]!=0:
                        roi_color[ey+i, ex+j]=glasses1[i,j]

        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (nx, ny, nw, nh) in nose:
            #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 127), 3)
            roi_nose = roi_gray[ny:ny + nh, nx:nx + nw]
            mustache1 = image_resize(mustache.copy(), width=nw)

            mw, mh, mc = mustache1.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    # print(glasses[i,j])
                    if mustache1[i, j][3] != 0:
                        roi_color[ny +int(nh/2.0) + i, nx + j] = mustache1[i, j]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
