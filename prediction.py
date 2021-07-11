import numpy as np
import cv2


def Predict(frame, tfnet, class_names, class_colors):
    frame = cv2.UMat.get(frame)
    predict = tfnet.return_predict(frame)
    tlx = 0
    brx = 0
    bry = 0
    tlxi = 0
    brxi = 0
    bryi = 0
    brxi_list = []
    brxi_list1 = []
    bryi_list = []
    bryi_list1 = []

    for item in predict:
        tlx = item['topleft']['x']
        tly = item['topleft']['y']
        brx = item['bottomright']['x']
        bry = item['bottomright']['y']
        label = item['label']
        conf = item['confidence']

        if conf > 0.3:

            for i in class_names:

                if label == i:
                    tlxi = tlx
                    bryi_list1.append(bry)
                    brxi_list1.append(brx)

                    class_num = class_names.index(i)
                    cv2.rectangle(frame, (tlx, tly), (brx, bry),
                                  class_colors[class_num], 2)
                    text = label + " " + ('%.2f' % conf)
                    cv2.rectangle(
                        frame, (tlx, tly - 15), (tlx + 100, tly + 5), class_colors[class_num], -1)
                    cv2.putText(frame, text, (tlx, tly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    break

        else:
            tlx = tlxi
            bry = bryi

    bryi_list = bryi_list + bryi_list1  # concatenate all lists
    brxi_list = brxi_list + brxi_list1
    if len(bryi_list) > 0:
        bryi = max(bryi_list)
        brxi = brxi_list[bryi_list.index(max(bryi_list))]
    return tlxi, brxi, bryi, frame


def transformPoints(M, x, y):
    point = np.array([x, y])
    homog_point = [point[0], point[1], 1]  # homog. coordinates
    transform_hp = np.array(np.dot(M, homog_point))  # transform
    transform_hp /= transform_hp[2]  # scale
    transformed_point = transform_hp[:2]  # remove cart. coor.

    return transformed_point[0], transformed_point[1]
