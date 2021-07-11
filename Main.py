from DefSet1 import region_of_interest
from DefSet2 import line_fit, viz2, final_viz2
from prediction import Predict, transformPoints
from darkflow.net.build import TFNet
import cv2
import numpy as np

# Define the codec and create VideoWriter object

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (1280, 662))

'''
Darkflow initialization
'''
options = {
    "model": "cfg/yolo.cfg",
    "load": "bin/yolo.weights",
    "threshold": 0.1,
    'gpu': 1.0}
tfnet = TFNet(options)
class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person', 'truck']

num_classes = len(class_names)
class_colors = []
for i in range(0, num_classes):
    hue = 255 * i / num_classes
    col = np.zeros((1, 1, 3)).astype("uint8")
    col[0][0][0] = hue
    col[0][0][1] = 128
    col[0][0][2] = 255
    cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
    col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
    class_colors.append(col)

'''
Loading the Video
'''
cap = cv2.VideoCapture('media/test0.mp4')


'''
Constants
'''
ret1 = {}

bottom_left = (10, 575)  # Short_Distance points
bottom_right = (1260, 575)
top_left = (450, 450)
top_right = (820, 450)

bottom_left1 = (0, 720)
top_left1 = (80, 0)
top_right1 = (575, 0)
bottom_right1 = (575, 720)

bottom_left2 = (575, 720)
top_left2 = (625, 0)
top_right2 = (625, 0)
bottom_right2 = (725, 720)

bottom_left3 = (675, 720)
top_left3 = (675, 0)
top_right3 = (1280, 0)
bottom_right3 = (1280, 720)
####
kernel_size = 5
low_threshold = 0
high_threshold = 100
region_of_interest_vertices = [
    bottom_left, top_left,
    top_right,
    bottom_right,
]

region_of_interest_vertices1 = [
    bottom_left1, top_left1,
    top_right1,
    bottom_right1,
]
region_of_interest_vertices2 = [
    bottom_left2, top_left2,
    top_right2,
    bottom_right2,
]

region_of_interest_vertices3 = [
    bottom_left3, top_left3,
    top_right3,
    bottom_right3,
]


'''
Main Framework
'''

i = 0
while(cap.isOpened()):
    i = i + 1
    frame=cv2.UMat(cap.read()[1])

    shape = frame.get().shape
###########################THRESHHOLDING#########################
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(frame, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 140, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    gauss_gray = cv2.GaussianBlur(mask_yw_image, (5, 5), 0)

    canny_edges = cv2.Canny(gauss_gray, low_threshold, high_threshold)

    cropped_image = region_of_interest(
        canny_edges, np.array(
            [region_of_interest_vertices], np.int32),shape)

####################### Prediction ######################################
    frame1 = region_of_interest(frame, np.array(
        [region_of_interest_vertices1], np.int32),shape)
    frame2 = region_of_interest(frame, np.array(
        [region_of_interest_vertices2], np.int32),shape)
    frame3 = region_of_interest(frame, np.array(
        [region_of_interest_vertices3], np.int32),shape)

    tlx_fst, brx_fsti, bry_fsti, frame1out = Predict(
        frame1, tfnet, class_names, class_colors)
    tlx_scnd, brx_scndi, bry_scndi, frame2out = Predict(
        frame2, tfnet, class_names, class_colors)
    tlx_thrd, brx_thrdi, bry_thrdi, frame3out = Predict(
        frame3, tfnet, class_names, class_colors)

############################Bird's Eye view (Perspective Transform)#######
    pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
    pts3 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])
    # Transformation Matrix for wraping
    M = cv2.getPerspectiveTransform(pts1, pts3)
    # Transformation Matrix for unwraping
    m_inv = cv2.getPerspectiveTransform(pts3, pts1)
    warped = cv2.warpPerspective(cropped_image, M, (1280, 720))

########################### transform points #############################
    brx_fst, bry_fst = transformPoints(M, brx_fsti + 80, bry_fsti)
    brx_scnd, bry_scnd = transformPoints(M, brx_scndi + 625, bry_scndi)
    brx_thrd, bry_thrd = transformPoints(M, brx_thrdi + 675, bry_thrdi)
    if bry_fst < 0 or bry_fst > 1157:
        bry_fst = 0
    if bry_scnd < 0 or bry_scnd > 1157:
        bry_scnd = 0
    if bry_thrd < 0 or bry_thrd > 1157:
        bry_thrd = 0

    ########################## Lane detection process ########################

    ret, sliding = line_fit(warped, bry_fst, bry_scnd,
                            bry_thrd, ret1, i)  # Line fitting
    ret1['middle1'] = ret['middle1_fit']
    ret1['middle2'] = ret['middle2_fit']
    ret1['test'] = ret['test']

    lineFit, left_fit, right_fit, middle1_fit, middle2_fit = viz2(
        warped, ret, bry_fst, bry_scnd, bry_thrd, save_file=None)  # Polyfit

    tlx_ignore, brx_ignore, _ignore, frame = Predict(
        frame, tfnet, class_names, class_colors)

    color_warp = final_viz2(
        frame,
        left_fit,
        right_fit,
        middle1_fit,
        middle2_fit,
        M,
        tlx_fst,
        tlx_scnd,
        tlx_thrd,
        bry_fst,
        bry_scnd,
        bry_thrd,
        ret)

    unwarped = cv2.warpPerspective(color_warp, m_inv, (1280, 720))

    result = cv2.addWeighted(frame, 1, unwarped, 0.35, 0)

    result3 = result[0:662, 0:1280]  # cropping unnecessary part of the video
    out.write(result3)  # Output video

    cv2.imshow('Result', result3)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    if cv2.waitKey(1) & 0xFF == ord('q') or i > 1000000:
        break


# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
