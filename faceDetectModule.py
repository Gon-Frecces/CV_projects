import cv2 as cv
import mediapipe as mp
import time


class Facedetector():
    def __init__(self, minDetectionCon = 0.5):
        
        self.minDetectionCon = minDetectionCon 

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()
        

    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                # cv.rectangle(img, bbox, (225, 0, 225), 2)
                cv.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2) 
        
        return img, bboxs
    
    def fancyDraw(self, img, bbox, l=30, t=150, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv.rectangle(img, bbox, (255, 0, 255), rt)
        # top left x,y 
        cv.line(img, (x,y), (x+l, y), (255, 0, 255), t)
        cv.line(img, (x,y), (x, y+l), (255, 0, 255), t)

        # top right x1,y 
        cv.line(img, (x1,y), (x1-l, y), (255, 0, 255), t)
        cv.line(img, (x1,y), (x1, y+l), (255, 0, 255), t)
        # bottom left x,y1 
        cv.line(img, (x,y1), (x+l, y1), (255, 0, 255), t)
        cv.line(img, (x,y1), (x, y1-l), (255, 0, 255), t)

        # bottom right x1,y1 
        cv.line(img, (x1,y1), (x1-l, y1), (255, 0, 255), t)
        cv.line(img, (x1,y1), (x1, y1+l), (255, 0, 255), t)


        return img


def main():
    cap = cv.VideoCapture('bigface.mp4')
    pTime = 0
    detector = Facedetector()
    width = 500
    height = 520
        
    while True:    
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)

        resized_frame = cv.resize(img,(width, height) )
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
        
        cv.putText(img, f'FPS: {int(fps)}', (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2) 
        cv.imshow('Image', resized_frame)
        cv.waitKey(1)



    

if __name__ == '__main__':
    main()