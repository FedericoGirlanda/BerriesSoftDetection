import random
import cv2
import numpy as np

"""
Plot bounding boxes from a prediction on the image img
"""

def plot_one_box(boxes, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = ((int(boxes[0]),int(boxes[1])),(int(boxes[2]),int(boxes[3])))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

"""
Get bounding box locations from raw output
"""

def getBoxes(predictions, dimension):
    boxes = np.array([int((predictions[0][0] - (0.5 * predictions[0][2]))*dimension[0]), 
                      int((predictions[0][1] - (0.5 * predictions[0][3]))*dimension[1]), 
                      int((predictions[0][0] + (0.5 * predictions[0][2]))*dimension[0]), 
                      int((predictions[0][1] + (0.5 * predictions[0][3]))*dimension[1]),
                      predictions[0][4]])
    for i in range(1,len(predictions)):
        boxes = np.vstack((boxes,[int((predictions[i][0] - (0.5 * predictions[i][2]))*dimension[0]), 
                                  int((predictions[i][1] - (0.5 * predictions[i][3]))*dimension[1]), 
                                  int((predictions[i][0] + (0.5 * predictions[i][2]))*dimension[0]), 
                                  int((predictions[i][1] + (0.5 * predictions[i][3]))*dimension[1]),
                                  predictions[i][4]]))
    return boxes

"""
Scale bounding box location for representation
"""

def scaleBox(p1,p2,w,h):
    return (int(p1[0]*w), int(p1[1]*h)), (int(p2[0]*w), int(p2[1]*h))

"""
Filter out boxes with low confidence 
"""

def filterBox(predictions, scoreThresh=0.5):
    return predictions[np.where(predictions.T[4]>scoreThresh)[0]]

"""
Filter out boxes that are significantly overlapped. Only preserve the box with the highest score.
This function should be called after filterBox to reduce workload
"""

def nonMaxSuppress(boxes, iouThresh=0.7):
    keep_idx = []
    sorted_indeces = np.array(boxes).T[4].argsort()
    boxes = np.array(boxes)[sorted_indeces]
    for i in range(len(boxes)):
        to_keep = True
        for j in range(i,len(boxes)):
            if box_iou(boxes[i], boxes[j]) > iouThresh and boxes[i][4] < boxes[j][4]:
                    to_keep = False
        if to_keep:
            keep_idx = np.append(keep_idx,i)
    return [boxes[int(k)] for k in keep_idx]

"""
Function that returns the IoU between two boxes
"""
def box_iou(box_a, box_b):

    def box_area(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    x_left = np.maximum(box_a[0], box_b[0])
    y_left = np.maximum(box_a[1], box_b[1])
    x_right = np.minimum(box_a[2], box_b[2])
    y_right = np.minimum(box_a[3], box_b[3])
    area_inter = max(0, x_right - x_left + 1) * max(0, y_right - y_left + 1) 
    area_union = float(area_a + area_b - area_inter) 

    return area_inter / area_union

"""
Function that calculate the centrid of a set of points
"""
def centroid(points):
    sum_x = 0
    sum_y = 0
    for i in range(len(points)):
        sum_x += points[i][0]
        sum_y += points[i][1]
    return (int(sum_x/len(points)), int(sum_y/len(points)))

"""
Function that calculate centers of boxes
"""
def boxesCenters(boxes):
    w0 = boxes[0][2] - boxes[0][0] 
    h0 = boxes[0][3] - boxes[0][1]
    centers = np.array([boxes[0][0]+w0/2,boxes[0][1]+h0/2])
    for i in range(1,len(boxes)):
        w = boxes[i][2] - boxes[i][0] 
        h = boxes[i][3] - boxes[i][1] 
        centers = np.vstack((centers, np.array([boxes[i][0]+w/2, boxes[i][1]+h/2])))
    return centers

"""
Function that implements a lowpass filter
"""
def lowpass(prev, next, beta):
    return beta*prev + (1-beta)*next

"""
Function that implements a series of erosions and dilations for removing outliers boxes centers, then it returna the rectangle that
enclose all the remaining shapes.
"""
def findBunch(centers):
    mask=np.zeros((640,640,1),np.uint8)
    for i in range(len(centers)):
        cv2.circle(mask, (int(centers[i][0]), int(centers[i][1])), radius=25, color=(255), thickness=-1)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=20)
    mask = cv2.dilate(mask, kernel, iterations=20)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    try:
        cnts = np.concatenate(cnts)
        x, y, w, h = cv2.boundingRect(cnts)
    except:
        x, y, w, h = 0,0,640,640

    # cv2.rectangle(mask, (x, y), (x + w - 1, y + h - 1), 255, 2)
    # cv2.imshow("out", mask)
    # cv2.waitKey(0)
    # assert False

    return x, y, h, w