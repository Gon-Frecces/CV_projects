import cv2
import time
import os
import handTrackModule as htm
import numpy as np
import pyautogui

wScr, hScr = pyautogui.size()
#####################
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
######################
plocX, plocY = 0, 0
clocX, clocY = 0, 0
print(wScr, hScr)
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(maxHands=1)
while True:
    # 1. find the hand landmarks
    # Import the image
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    # 2. get the tip of index and midle fingers
    if len(lmList) != 0:
        x1,y1 = lmList[0][1:]
        x2,y2 = lmList[12][1:]
        print(x1, y1, x2, y2)
        # 3. Check which of the fingers are up

        fingers = detector.fingersUp()
        print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam -frameR, hCam-frameR), (255,0, 255), 2)
        # 4. Only Index finger : moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            
            
            # (moving mode)
            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (frameR,hCam-frameR), (0,hScr))

            # 6. Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocX + (y3 - plocY) / smoothening

            # 7. move mouse
            pyautogui.moveTo(wScr-x3, y3)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # 8. both index and middle fingers are up: Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 9. Find the distance between fingers
            # 10. Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

                pyautogui.click(lineInfo[4], lineInfo[5])
        

    # 11. Frame rate

    

        cTime = time.time()
        fps= 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
 

    cv2.imshow('Image', img)
  
    cv2.waitKey(1)