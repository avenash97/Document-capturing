import cv2
import numpy as np
import sys
import getopt
import operator
from numpy import linalg as LA

def readImage(filename):
    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successfully read...')
        return img

def findCorners(img, window_size, k, thresh):
    dy, dx = np.gradient(img)
    print(dy.shape,dx.shape)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]

    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = int(window_size/2)

    print("Finding Corners...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
           
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()
            A = np.array([[Sxx, Sxy], [Sxy, Syy]])

            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            #r = det - k*(trace**2)     // part 2

            eig = LA.eigvals(A)
            #r = eig[0]*eig[1] -k*(eig[0]+eig[1])*(eig[0]+eig[1]) // part 3
            r = min(eig[0],eig[1])   # part 1
            if r > thresh:
                #print(x, y, r)
                cornerList.append([x, y, r])
                # color_img.itemset((y, x, 0), 0)
                # color_img.itemset((y, x, 1), 0)
                # color_img.itemset((y, x, 2), 255)
    return color_img, cornerList

def main():
    window_size = 11
    k = 0.04
    thresh = 50000

    img = cv2.imread("check.jpg")
    blur = cv2.blur(img,(50,50))
    blur= cv2.copyMakeBorder(blur,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,255,0])
    #img = cv2.resize(img,(250,250))
    width,height,_ = img.shape
    
    if img is not None:
        # if len(img.shape) == 3:
        grey = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        # if len(img.shape) == 4:
            # grey = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        dst = cv2.cornerHarris(grey,2,3,0.04)

        #result is dilated for marking the corners, not important
        # dst = cv2.dilate(dst,None)
        t = dst.argsort(axis=None)[-4:][::-1]
        print(t)
        x = t%height
        y= t%width
        # Threshold for an optimal value, it may vary depending on the image.
        blur[dst>0.01*dst.max()]=[0,0,255]
        # finalImg, cornerList = findCorners(img, int(window_size), float(k), int(thresh))

        for i in range(4):
            # print(x)
            cv2.circle(img,(x[i],y[i]),30,(0,0,255),-1)
        # if finalImg is not None:
        print(dst.shape)
        cv2.imwrite("blur.png",blur)
        cv2.imwrite("harris.png", img)
        

if __name__ == "__main__":
    main()
