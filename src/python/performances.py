import numpy as np
import cv2
import pandas as pd

from utils import plot_one_box, box_iou

IoU_thresh = 0.5
bunchDetection = False

# Create tests result file if not present
results_csv = "results/detectionBoxes.csv"
test_csv = "dataset/EIdataset/testing/testingBoxes.csv"
data_test = pd.read_csv(test_csv)
data_results = pd.read_csv(results_csv)

# Testing all the test image
idx_test = []
idx_results = []
for i in range(len(data_test)):
    try:
        cv2.imread("dataset/EIdataset/testing/"+data_test.image[i])
        idx_test = np.append(idx_test,i)
        idx_results = np.append(idx_results,np.where(data_results.image == data_test.image[i])[0][0])
    except:
        pass

dim = (640,640)
TP_bunch = 0
totAcc_berries = 0
for j in range(len(idx_test)):
    image = cv2.imread("dataset/EIdataset/testing/"+data_test.image[idx_test[j]])
    image = cv2.resize(image, dim, interpolation = cv2.INTER_NEAREST)
    if not bunchDetection:
        if j < len(idx_test)-1:
            n_boxesTest = idx_test[j+1]-idx_test[j]
            n_boxesRes = idx_results[j+1]-idx_results[j]
        else:
            n_boxesTest = (len(data_test.boxes0)-1) -idx_test[j]
            n_boxesRes = (len(data_results.boxes0)-1)-idx_results[j]

        # Plot the boxes
        # plot test boxes
        for b in range(1,int(n_boxesTest)):
            box = np.array([data_test.boxes0[idx_test[j]+b]*dim[0]-(data_test.boxes2[idx_test[j]+b]*dim[0]/2),
                            data_test.boxes1[idx_test[j]+b]*dim[1]-(data_test.boxes3[idx_test[j]+b]*dim[1]/2),
                            data_test.boxes0[idx_test[j]+b]*dim[0]+data_test.boxes2[idx_test[j]+b]*dim[0]/2,
                            data_test.boxes1[idx_test[j]+b]*dim[1]+data_test.boxes3[idx_test[j]+b]*dim[1]/2])
            plot_one_box(box, image, (0,0,255), "berry", 1)
        # plot results boxes
        for b in range(1,int(n_boxesRes)):
            box = np.array([data_results.boxes0[idx_results[j]+b],
                            data_results.boxes1[idx_results[j]+b],
                            data_results.boxes2[idx_results[j]+b],
                            data_results.boxes3[idx_results[j]+b]])
            plot_one_box(box, image, (255,0,0), "berry", 1)

        # Berries detection accuracy measure
        TP = 0
        FP = 0
        accuracy = 0
        for bt in range(1,int(n_boxesTest)):
            for br in range(1,int(n_boxesRes)):
                box_t = np.array([data_test.boxes0[idx_test[j]+bt]*dim[0]-(data_test.boxes2[idx_test[j]+bt]*dim[0]/2),
                                  data_test.boxes1[idx_test[j]+bt]*dim[1]-(data_test.boxes3[idx_test[j]+bt]*dim[1]/2),
                                  data_test.boxes0[idx_test[j]+bt]*dim[0]+data_test.boxes2[idx_test[j]+bt]*dim[0]/2,
                                  data_test.boxes1[idx_test[j]+bt]*dim[1]+data_test.boxes3[idx_test[j]+bt]*dim[1]/2])
                box_r = np.array([data_results.boxes0[idx_results[j]+br],
                                data_results.boxes1[idx_results[j]+br],
                                data_results.boxes2[idx_results[j]+br],
                                data_results.boxes3[idx_results[j]+br]])
                iou = box_iou(box_t,box_r)
                if iou < IoU_thresh:
                    FP += 1
                else:
                    TP += 1
        accuracy = TP/(TP+FP)
        print(f"Image {j} has an accuracy of {accuracy}")
        totAcc_berries += accuracy

    else:
        # Plot the boxes
        box_t = np.array([data_test.boxes0[idx_test[j]],
                            data_test.boxes1[idx_test[j]],
                            data_test.boxes2[idx_test[j]],
                            data_test.boxes3[idx_test[j]]])
        box_r = np.array([data_results.boxes0[idx_results[j]],
                        data_results.boxes1[idx_results[j]],
                        data_results.boxes2[idx_results[j]],
                        data_results.boxes3[idx_results[j]]])
        # plot test box
        plot_one_box(box_t, image, (0,0,255), "berry", 1)
        # plot results box
        plot_one_box(box_r, image, (255,0,0), "berry", 1)

        # Bunch detection accuracy measure
        iou = box_iou(box_t,box_r)
        print(f"Image {j} has an IoU of {iou}")
        if iou > IoU_thresh:
            TP_bunch += 1
    
    cv2.imshow("Output", image)
    cv2.waitKey(0)

print("")
if TP_bunch > 0:
    print(f"The accuracy of the bunch detection is {TP_bunch/len(idx_test)}")
if totAcc_berries > 0:
    print(f"The average accuracy of the berry detection is {totAcc_berries/len(idx_test)}")