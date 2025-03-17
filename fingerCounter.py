import cv2
import time
import os
import handTrackModule as htm
wCam, hCam = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = 'FingerImages'
myList  = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]   # tips ofeach finger

while True:
    success, img = cap.read()
    img1 = detector.findHands(img)
    lmList = detector.findPosition(img1, draw=False)
    print(lmList)

    if lmList is not None and len(lmList) != 0:
        fingers = []

        # Thumbs na special case it's closed on the right open on the left (right hand and viceversa for LH)
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            print('Index finger open')    # aim here is to know which finger landmarks are closed 
            fingers.append(1)  #finger open       # and which are released(using the comparison operator on their position)
        else:
            fingers.append(0)    

        # rest of 4 fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                print('Index finger open')    # aim here is to know which finger landmarks are closed 
                fingers.append(1)  #finger open       # and which are released(using the comparison operator on their position)
            else:
                fingers.append(0)   # finger closed
        print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers-1].shape
        img1[0:h, 0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img1, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img1, str(totalFingers), (43, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
 

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img1, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
   
    cv2.imshow('Image', img1)
    cv2.waitKey(1)