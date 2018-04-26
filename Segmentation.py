import cv2
import numpy as np
import glob
import cv2
import glob
class Coordinates:
    def __init__(self,x,y,w,h):
     self.x=x
     self.y=y
     self.w=w
     self.h=h
class Segmentation:
 def SegmentAll(self):
  for imgs in glob.glob("F:/4/sem1/Pattern/Tasks/Pattern Project/Testing/*.png"):
    ActuallImage=cv2.imread(imgs.split('.')[0]+'.jpg')
    Objects=self.Segment(imgs)
    for Object in range(0,len(Objects),1):
     crop_img = ActuallImage[Objects[Object].y:Objects[Object].y + Objects[Object].h, Objects[Object].x:Objects[Object].x + Objects[Object].w]
     cv2.imwrite(imgs+str(Object)+'.bmp', crop_img, params=None)
 def Segment(self,Path):
     Image = cv2.imread(Path)
     gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
     mx = np.max(gray)
     size = [0, 0]
     size[0] = len(gray)
     size[1] = len(gray[0])
     Objects=[]
     while (mx):
         ret, thresh = cv2.threshold(gray, mx - 6, mx, 0)
         for i in range(0, size[0], 1):
             for j in range(0, size[1], 1):
                 if (mx - gray[i][j] < 6):
                     gray[i][j] = 0
         mx = np.max(gray)
         im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
         idx = 0
         mx_area = 0
         for contour in range(0, len(contours), 1):
             area = cv2.contourArea(contours[contour])
             if (mx_area < area):
                 mx_area = area
                 idx = contour
         if (mx_area < 300):
             continue
         x, y, w, h = cv2.boundingRect(contours[idx])
         Objects.append(Coordinates(x,y,w,h))
     return Objects
         # cv2.drawContours(img, contours[idx], 0, (0,255,0), 3)
         # cv2.imshow('out',img)