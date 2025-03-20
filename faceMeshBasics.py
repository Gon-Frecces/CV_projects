import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('bigface.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
width = 500
height = 520

while True:    
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                  drawSpec, drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)
    # print(results)
          
    
   
    resized_frame = cv.resize(img,(width, height) )
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
 
    
    cv.putText(img, f'FPS: {int(fps)}', (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2) 
    cv.imshow('Image', resized_frame)
    cv.waitKey(1)