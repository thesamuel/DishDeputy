import cv2
import numpy as np
import json
import io

#TODO: make global?
pts = []

# mouse callback function
def add_point(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        pts.append([x,y])
    elif event == cv2.EVENT_FLAG_RBUTTON:
        if not not pts:
            pts.pop()

def make_mask(height, width, pts):
    # TODO: save as nparray
    mmask = np.zeros((int(width),int(height), 3), np.uint8)
    mmask = cv2.bitwise_not(mmask)
    npts = np.asarray(pts)
    npts = npts.reshape(-1,1,2)
    cv2.fillConvexPoly(mmask,npts,0)
    return mmask

def choose_points():
    cv2.namedWindow('chooser')
    cap = cv2.VideoCapture(0)
    height = cap.get(3)
    width = cap.get(4)
    print("capture res", height, width)
    
    while(True):
        # Get next frame
        ret, frame = cap.read()
            
        if not not pts: 
            mask = make_mask(height, width, pts) 
            frame = cv2.addWeighted(frame,0.7,mask,0.3,0)
            for pt in pts:
                cv2.circle(frame,(pt[0],pt[1]), 2, (0,0,255), -1)
    
        cv2.imshow('chooser',frame)
        # TODO: only add listener when window closed
        cv2.setMouseCallback('chooser',add_point)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            data = {'height':height, 'width':width, 'points':pts}
            with open('data.txt', 'w') as outfile:
                json.dump(data, outfile)
    
    
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
def watch():
    #TODO: exceptions
    #TODO: promt user for different files
    with open('data.txt') as data_file:
        loaded = json.load(data_file)
         
    print(loaded.get('points'))

    watch_mask = make_mask(loaded.get('height'), loaded.get('width'), loaded.get('points'))
    watch_mask = cv2.cvtColor(watch_mask, cv2.COLOR_BGR2GRAY)
    watch_mask = cv2.bitwise_not(watch_mask)
    cv2.namedWindow('watcher')
    cap = cv2.VideoCapture(0)

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml') 
    eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

    # TODO: set timelimits, etc.
    while(True):
        ret, frame = cap.read()
        
        #TODO: crop after background subtraction?
        #TODO: try different background subtractions
        fgmask = fgbg.apply(frame)

        cropped = cv2.bitwise_and(fgmask, fgmask, mask = watch_mask)
        
        fgmask_b = cv2.GaussianBlur(cropped, (21, 21), 0)

        #gray = cv2.cvtColor(fgmask_b, cv2.COLOR_BGR2GRAY)

        ret,thresh = cv2.threshold(fgmask_b,127,255,0)

        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion = False
        for c in contours:
            if cv2.contourArea(c) < 10:
                continue
            motion = True
            (x, y, w, h) = cv2.boundingRect(c) 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if motion:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)


        full = cv2.cvtColor(fgmask_b, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(full, contours, -1, (0,0,255), 2) 

        stacked = np.hstack((full, frame))
        cv2.imshow('watcher', stacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#TODO: add choices
choose_points()
watch()
