import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from ocr.helpers import implt, resize, ratio
plt.rcParams['figure.figsize'] = (9.0, 9.0)

IMG = "check.jpg" 
image = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2RGB)
implt(image)

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




cv2.imwrite(IMG, cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB))
