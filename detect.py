import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from ocr.helpers import implt, resize, ratio
plt.rcParams['figure.figsize'] = (12.0, 12.0)

IMG = "check2.jpg" 
image = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
implt(image)


def fourCornersSort(pts):
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])

def contourOffset(cnt, offset):
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt


def findPageContours(edges, img):
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * 0.5
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    maxArea = MIN_COUNTOUR_AREA
    pageContour = np.array([[0, 0],
                            [0, height-5],
                            [width-5, height-5],
                            [width-5, 0]])

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                maxArea < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
            
            maxArea = cv2.contourArea(approx)
            pageContour = approx[:, 0]

    pageContour = fourCornersSort(pageContour)
    return contourOffset(pageContour, (-5, -5))
    
pageContour = findPageContours(closedEdges, resize(image))
print("PAGE CONTOUR:")
print(pageContour)
implt(cv2.drawContours(resize(image), [pageContour], -1, (0, 255, 0), 3))

pageContour = pageContour.dot(ratio(image))

def edgesDet(img, minVal, maxVal):

    img = cv2.cvtColor(resize(img), cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    implt(img, 'gray', 'Adaptive Threshold')

    img = cv2.medianBlur(img, 11)

    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    implt(img, 'gray', 'Median Blur + Border')

    return cv2.Canny(img, minVal, maxVal)
    
imageEdges = edgesDet(image, 200, 250)

closedEdges = cv2.morphologyEx(imageEdges, cv2.MORPH_CLOSE, np.ones((5, 11)))
implt(closedEdges, 'gray', 'Edges')


def harris(img,window_size,alpha,threshold):
    """
    Score R:= product(eigenvalues) - alpha*(sum(eigenvalues))
    """
    start = time.process_time()
    print("Harris Corner Detection")
    coordinate_y = []
    coordinate_x = []
    strided = windows_strided(img,window_size)
    k = window_size//2
    l = window_size//2
    c = 0
    for stride in strided:
        for i in stride:
            temp = eigenvals(i)
            R = np.subtract(np.prod(temp),np.multiply(alpha,np.sum(temp)))
            # if R>0:
            #     print(R)
            if R > threshold:
                c+=1
                coordinate_x.append(k)
                coordinate_y.append(l)
            k+=1
        l+=1
        k=window_size//2
    print("Corners Detected:{}".format(c))
    print("Total time: {}s".format(time.process_time()- start))
    return [coordinate_x,coordinate_y]

def perspImageTransform(img, sPoints):
    height = max(np.linalg.norm(sPoints[0] - sPoints[1]),
                 np.linalg.norm(sPoints[2] - sPoints[3]))
    width = max(np.linalg.norm(sPoints[1] - sPoints[2]),
                 np.linalg.norm(sPoints[3] - sPoints[0]))
    
    tPoints = np.array([[0, 0],
                        [0, height],
                        [width, height],
                        [width, 0]], np.float32)
    
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)
    
    M = cv2.getPerspectiveTransform(sPoints, tPoints) 
    return cv2.warpPerspective(img, M, (int(width), int(height)))
    
    
newImage = perspImageTransform(image, pageContour)
implt(newImage, t='Result')

cv2.imwrite(IMG, cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))
