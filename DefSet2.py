import numpy as np
import cv2
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


def line_fit(binary_warped, bry_fst, bry_scnd, bry_thrd, ret1, i):
    """
    Find and fit lane lines
    """
    binary_warped = cv2.UMat.get(binary_warped)
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = (
        np.dstack(
            (binary_warped,
             binary_warped,
             binary_warped)) *
        255).astype('uint8')
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    leftx_base = np.argmax(histogram[60:250]) + 60
    middle1_base = np.argmax(histogram[425:560]) + 425
    middle2_base = np.argmax(histogram[735:815]) + 735
    rightx_base = np.argmax(histogram[1050:1250]) + 1050

    bry_fstN = bry_fst

    bry_scndN = bry_scnd

    bry_thrdN = bry_thrd

    # Choose the number of sliding windows
    nwindows = 50
    # Set height of windows

    window_height_left = np.int((binary_warped.shape[0] - bry_fstN) / nwindows)
    window_height_right = np.int(
        (binary_warped.shape[0] - bry_thrdN) / nwindows)
    window_height_middle = np.int(
        (binary_warped.shape[0] - bry_scndN) / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()

    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    # Main positions to be updated for each window
    leftx_current = leftx_base
    middle1_current = middle1_base
    middle2_current = middle2_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    left_right_margin = 45
    middle_margin = 25
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left,middle, and right lane pixel indices
    left_lane_inds = []
    middle1_lane_inds = []
    middle2_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low_left = binary_warped.shape[0] - \
            (window + 1) * window_height_left
        win_y_high_left = binary_warped.shape[0] - window * window_height_left

        win_y_low_right = binary_warped.shape[0] - \
            (window + 1) * window_height_right
        win_y_high_right = binary_warped.shape[0] - \
            window * window_height_right

        win_y_low_middle = binary_warped.shape[0] - \
            (window + 1) * window_height_middle
        win_y_high_middle = binary_warped.shape[0] - \
            window * window_height_middle

        ###

        win_xleft_low = leftx_current - left_right_margin
        win_xleft_high = leftx_current + left_right_margin

        win_xmiddle1_low = middle1_current - middle_margin
        win_xmiddle1_high = middle1_current + middle_margin

        win_xmiddle2_low = middle2_current - middle_margin
        win_xmiddle2_high = middle2_current + middle_margin

        win_xright_low = rightx_current - left_right_margin
        win_xright_high = rightx_current + left_right_margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low_left),
                      (win_xleft_high, win_y_high_left), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xmiddle1_low, win_y_low_middle),
                      (win_xmiddle1_high, win_y_high_middle), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xmiddle2_low, win_y_low_middle),
                      (win_xmiddle2_high, win_y_high_middle), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low_right),
                      (win_xright_high, win_y_high_right), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = (
            (nonzeroy >= win_y_low_left) & (
                nonzeroy < win_y_high_left) & (
                nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
        good_middle1_inds = (
            (nonzeroy >= win_y_low_middle) & (
                nonzeroy < win_y_high_middle) & (
                nonzerox >= win_xmiddle1_low) & (
                nonzerox < win_xmiddle1_high)).nonzero()[0]
        good_middle2_inds = (
            (nonzeroy >= win_y_low_middle) & (
                nonzeroy < win_y_high_middle) & (
                nonzerox >= win_xmiddle2_low) & (
                nonzerox < win_xmiddle2_high)).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low_right) & (
                nonzeroy < win_y_high_right) & (
                nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        middle1_lane_inds.append(good_middle1_inds)
        middle2_lane_inds.append(good_middle2_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean
        # position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_middle1_inds) > minpix:
            middle1_current = np.int(np.mean(nonzerox[good_middle1_inds]))
        if len(good_middle2_inds) > minpix:
            middle2_current = np.int(np.mean(nonzerox[good_middle2_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    middle1_lane_inds = np.concatenate(middle1_lane_inds)
    middle2_lane_inds = np.concatenate(middle2_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left, middle, and right line pixel positions

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]

    middle1x = nonzerox[middle1_lane_inds]
    middle1y = nonzeroy[middle1_lane_inds]

    middle2x = nonzerox[middle2_lane_inds]
    middle2y = nonzeroy[middle2_lane_inds]

    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    # Return a dict of relevant variables
    ret = {}

    if len(leftx) > 0 and len(lefty) > 0 and (
        bry_fst <= 720) and (
        bry_fst >= 0) and (
            lefty.shape[0] > 2000):
        left_fit = np.polyfit(lefty, leftx, 2)
        ret['left_fit'] = left_fit
    else:
        ret['left_fit'] = 'error'
    if len(rightx) > 0 and len(righty) > 0 and (
        bry_thrd <= 720) and (
        bry_thrd >= 0) and (
            righty.shape[0] > 2000):
        right_fit = np.polyfit(righty, rightx, 2)
        ret['right_fit'] = right_fit
    else:
        ret['right_fit'] = 'error'

    if len(middle2x) > 0 and len(middle2y) > 0 and (
        bry_scnd <= 720) and (
        bry_scnd >= 0) and (
            middle2y.shape[0] > 100):
        if (max(middle2y) > 600 or max(middle2y) == 482):
            middle2_fit = np.polyfit(middle2y, middle2x, 2)
            ret['middle2_fit'] = middle2_fit
        else:
            if i > 1:
                ret['middle2_fit'] = ret1['middle2']
            else:
                ret['middle2_fit'] = 'error'
    else:
        if i > 1:
            ret['middle2_fit'] = ret1['middle2']
        else:
            ret['middle2_fit'] = 'error'

    if len(middle1x) > 0 and len(middle1y) > 0 and (
        bry_scnd <= 720) and (
        bry_scnd >= 0) and (
            middle1y.shape[0] > 100):
        if (max(middle1y) > 600 or max(middle1y) == 482):
            middle1_fit = np.polyfit(middle1y, middle1x, 2)
            ret['middle1_fit'] = middle1_fit
        else:
            if i > 1:
                ret['middle1_fit'] = ret1['middle1']
            else:
                ret['middle1_fit'] = 'error'
    else:
        if i > 1:
            ret['middle1_fit'] = ret1['middle1']
        else:
            ret['middle1_fit'] = 'error'

    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['out_img'] = out_img
    ret['left_lane_inds'] = left_lane_inds
    ret['middle1_lane_inds'] = middle1_lane_inds
    ret['middle2_lane_inds'] = middle2_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    ret['left_lane_nonzerox'] = np.count_nonzero(histogram[60:250])
    ret['middle1_lane_nonzerox'] = np.count_nonzero(histogram[425:560])
    ret['middle2_lane_nonzerox'] = np.count_nonzero(histogram[735:815])
    ret['right_lane_nonzerox'] = np.count_nonzero(histogram[1050:1250])

    ret['left_lane_ymax'] = lefty.shape[0]
    ret['middle1_lane_ymax'] = middle1y.shape[0]
    ret['middle2_lane_ymax'] = middle2y.shape[0]
    ret['right_lane_ymax'] = righty.shape[0]

    ret['test'] = middle1_base

    return ret, out_img


def viz2(binary_warped, ret, bry_fst, bry_scnd, bry_thrd, save_file=None):
    """
    Visualize the predicted lane lines with margin, on binary warped image
    save_file is a string representing where to save the image (if None, then just display)
    """
    # Grab variables from ret dictionary
    binary_warped = cv2.UMat.get(binary_warped)
    left_fit = ret['left_fit']
    middle1_fit = ret['middle1_fit']
    middle2_fit = ret['middle2_fit']
    right_fit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    left_lane_inds = ret['left_lane_inds']
    middle1_lane_inds = ret['middle1_lane_inds']
    middle2_lane_inds = ret['middle2_lane_inds']
    right_lane_inds = ret['right_lane_inds']

    # Create an image to draw on and an image to show the selection window
    out_img = (
        np.dstack(
            (binary_warped,
             binary_warped,
             binary_warped)) *
        255).astype('uint8')
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[middle1_lane_inds],
            nonzerox[middle1_lane_inds]] = [0, 255, 0]
    out_img[nonzeroy[middle2_lane_inds],
            nonzerox[middle2_lane_inds]] = [0, 255, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate x and y values for plotting

    ploty_left = np.linspace(
        binary_warped.shape[0] - 1,
        abs(bry_fst),
        binary_warped.shape[0])
    ploty_middle = np.linspace(
        binary_warped.shape[0] - 1,
        abs(bry_scnd),
        binary_warped.shape[0])
    ploty_right = np.linspace(
        binary_warped.shape[0] - 1,
        abs(bry_thrd),
        binary_warped.shape[0])
    if (ret['left_fit'] != 'error'):
        left_fitx = left_fit[0] * ploty_left**2 + \
            left_fit[1] * ploty_left + left_fit[2]
    if (ret['middle1_fit'] != 'error'):
        middle1_fitx = middle1_fit[0] * ploty_middle**2 + \
            middle1_fit[1] * ploty_middle + middle1_fit[2]

    if (ret['middle2_fit'] != 'error'):
        middle2_fitx = middle2_fit[0] * ploty_middle**2 + \
            middle2_fit[1] * ploty_middle + middle2_fit[2]

    if (ret['right_fit'] != 'error'):
        right_fitx = right_fit[0] * ploty_right**2 + \
            right_fit[1] * ploty_right + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    middle_margin = 25
    left_right_margin = 60  # NOTE: Keep this in sync with *_fit()

    if (ret['left_fit'] != 'error'):
        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx - left_right_margin, ploty_left]))])
        left_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([left_fitx + left_right_margin, ploty_left])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))

    if (ret['middle1_fit'] != 'error'):
        middle1_line_window1 = np.array(
            [np.transpose(np.vstack([middle1_fitx - middle_margin, ploty_middle]))])
        middle1_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([middle1_fitx + middle_margin, ploty_middle])))])
        middle1_line_pts = np.hstack(
            (middle1_line_window1, middle1_line_window2))
        cv2.fillPoly(window_img, np.int_([middle1_line_pts]), (0, 255, 0))

    if (ret['middle2_fit'] != 'error'):
        middle2_line_window1 = np.array(
            [np.transpose(np.vstack([middle2_fitx - middle_margin, ploty_middle]))])
        middle2_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([middle2_fitx + middle_margin, ploty_middle])))])
        middle2_line_pts = np.hstack(
            (middle2_line_window1, middle2_line_window2))
        cv2.fillPoly(window_img, np.int_([middle2_line_pts]), (0, 255, 0))

    if (ret['right_fit'] != 'error'):
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx - left_right_margin, ploty_right]))])
        right_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([right_fitx + left_right_margin, ploty_right])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    return result, left_fit, right_fit, middle1_fit, middle2_fit


def final_viz2(
        undist,
        left_fit,
        right_fit,
        middle1_fit,
        middle2_fit,
        m_inv,
        tlx_fst,
        tlx_scnd,
        tlx_thrd,
        bry_fst,
        bry_scnd,
        bry_thrd,
        ret):
    # def final_viz2(undist, left_fit, right_fit, middle1_fit, middle2_fit,
    # m_inv, left_curve, right_curve, vehicle_offset)
    """
    Final lane line prediction visualized and overlayed on top of original image
    """

    # Generate x and y values for plotting
    ploty_left = np.linspace(
        undist.shape[0] - 1,
        abs(bry_fst),
        undist.shape[0])
    ploty_middle = np.linspace(
        undist.shape[0] - 1,
        abs(bry_scnd),
        undist.shape[0])
    ploty_right = np.linspace(
        undist.shape[0] - 1,
        abs(bry_thrd),
        undist.shape[0])

    if (left_fit != 'error'):
        left_fitx = left_fit[0] * ploty_left**2 + \
            left_fit[1] * ploty_left + left_fit[2]
    if (right_fit != 'error'):
        right_fitx = right_fit[0] * ploty_right**2 + \
            right_fit[1] * ploty_right + right_fit[2]
    if (middle1_fit != 'error'):
        middle1_fitx = middle1_fit[0] * ploty_middle**2 + \
            middle1_fit[1] * ploty_middle + middle1_fit[2]
    if (middle2_fit != 'error'):
        middle2_fitx = middle2_fit[0] * ploty_middle**2 + \
            middle2_fit[1] * ploty_middle + middle2_fit[2]

    # Create an image to draw the lines on
    #warp_zero = np.zeros_like(warped).astype(np.uint8)
    #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # NOTE: Hard-coded image dimensions
    color_warp = np.zeros((720, 1280, 3), dtype='uint8')

    # Recast the x and y points into usable format for cv2.fillPoly()

    # Draw the lane onto the warped blank image

# ((((((((((((()))))))))))))))))))))))))))++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (right_fit != 'error') and (middle2_fit != 'error') and (
            ret['middle2_lane_ymax'] > 100) and (ret['right_lane_ymax'] > 2000):
        slow = cv2.imread('media/signs/Webp.net-resizeimage.png')
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty_right])))])
        pts_middle2 = np.array(
            [np.transpose(np.vstack([middle2_fitx, ploty_middle]))])
        avg_middle2 = int(sum(middle2_fitx) / len(middle2_fitx))
        avg_right = int(sum(right_fitx) / len(right_fitx))
        avg_thrd = int((avg_middle2 + avg_right) / 2)

        if (pts_right.min() < pts_middle2.min()):
            pts_right[pts_right < pts_middle2.min()] = pts_middle2.min()
        if (pts_middle2.min() < pts_right.min()):
            pts_middle2[pts_middle2 < pts_right.min()] = pts_right.min()
        pts3 = np.hstack((pts_middle2, pts_right))

        if (ret['middle2_lane_ymax'] > 7000):
            cv2.putText(
                undist,
                'Right lane is prohibited !',
                (50,
                 150),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0,
                 255,
                 255),
                1,
                lineType=cv2.LINE_AA)
            cv2.fillPoly(color_warp, np.int_([pts3]), (0, 0, 0))
        else:
            if (tlx_thrd > 0) and (bry_thrd >= 0) and (bry_thrd <= 720):
                cv2.putText(
                    undist,
                    'Be careful ! There is a vehicle on the right lane.',
                    (50,
                     150),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0,
                     255,
                     255),
                    1,
                    lineType=cv2.LINE_AA)
                cv2.fillPoly(
                    color_warp,
                    np.int_(
                        [pts3]),
                    ((-255 / 720) * bry_thrd + 255,
                     (-255 / 720) * bry_thrd + 255,
                        ((255 / 720) * bry_thrd)))
                cv2.putText(
                    color_warp,
                    'Right',
                    (avg_thrd - 50,
                     700),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.75,
                    (0,
                     0,
                     0),
                    5,
                    lineType=cv2.LINE_AA)

                if (int(pts_right.min()) >= 500) and (avg_thrd > 75):
                    slow = cv2.addWeighted(color_warp[int(pts_right.min()):650, (avg_thrd - 75):(
                        avg_thrd + 75)], 1, slow[(int(pts_right.min()) - 500):150, 0:150], 1, 0)
                    color_warp[int(pts_right.min()):650,
                               (avg_thrd - 75):(avg_thrd + 75)] = slow
                else:
                    if (avg_thrd > 75):
                        slow = cv2.addWeighted(
                            color_warp[500:650, (avg_thrd - 75):(avg_thrd + 75)], 1, slow[0:150, 0:150], 1, 0)
                        color_warp[500:650,
                                   (avg_thrd - 75):(avg_thrd + 75)] = slow

            else:
                if (tlx_thrd == 0):
                    cv2.fillPoly(
                        color_warp,
                        np.int_(
                            [pts3]),
                        ((-255 / 720) * bry_thrd + 255,
                         (-255 / 720) * bry_thrd + 255,
                            ((255 / 720) * bry_thrd)))
                    cv2.putText(
                        color_warp,
                        'Right',
                        (avg_thrd - 50,
                         700),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.75,
                        (0,
                         0,
                         0),
                        5,
                        lineType=cv2.LINE_AA)
                else:
                    cv2.fillPoly(
                        color_warp,
                        np.int_(
                            [pts3]),
                        ((-255 / 720) * bry_thrd + 255,
                         (-255 / 720) * bry_thrd + 255,
                            ((255 / 720) * bry_thrd)))
                    cv2.putText(
                        color_warp,
                        'Right',
                        (avg_thrd - 50,
                         700),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.75,
                        (0,
                         0,
                         0),
                        5,
                        lineType=cv2.LINE_AA)
                    if (int(pts_right.min()) >= 500):
                        slow = cv2.addWeighted(color_warp[int(pts_right.min()):650, (avg_thrd - 75):(
                            avg_thrd + 75)], 1, slow[(int(pts_right.min()) - 500):150, 0:150], 1, 0)
                        color_warp[int(pts_right.min()):650,
                                   (avg_thrd - 75):(avg_thrd + 75)] = slow
                    else:
                        slow = cv2.addWeighted(
                            color_warp[500:650, (avg_thrd - 75):(avg_thrd + 75)], 1, slow[0:150, 0:150], 1, 0)
                        color_warp[500:650,
                                   (avg_thrd - 75):(avg_thrd + 75)] = slow
                    cv2.putText(
                        undist,
                        'Be careful ! There is a vehicle on the right lane.',
                        (50,
                         150),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.5,
                        (0,
                         255,
                         255),
                        1,
                        lineType=cv2.LINE_AA)
    else:
        if ((ret['middle2_lane_ymax'] > 7000)):
            cv2.putText(
                undist,
                'Right lane is prohibited !',
                (50,
                 150),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0,
                 255,
                 255),
                1,
                lineType=cv2.LINE_AA)
        else:

            if (tlx_thrd > 0):
                cv2.putText(
                    undist,
                    'Be careful ! There is a vehicle on the right lane.',
                    (50,
                     150),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0,
                     255,
                     255),
                    1,
                    lineType=cv2.LINE_AA)
            else:
                cv2.putText(
                    undist,
                    'Right lane is not accessible !',
                    (50,
                     150),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0,
                     255,
                     255),
                    1,
                    lineType=cv2.LINE_AA)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if (left_fit != 'error') and (middle1_fit != 'error') and (
            ret['middle1_lane_ymax'] > 100) and (ret['left_lane_ymax'] > 2000):
        slow = cv2.imread('media/signs/Webp.net-resizeimage.png')
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty_left]))])
        pts_middle1 = np.array(
            [np.flipud(np.transpose(np.vstack([middle1_fitx, ploty_middle])))])
        avg_middle1 = int(sum(middle1_fitx) / len(middle1_fitx))
        avg_left = int(sum(left_fitx) / len(left_fitx))
        avg_fst = int((avg_middle1 + avg_left) / 2)
        if (pts_left.min() < pts_middle1.min()):
            pts_left[pts_left < pts_middle1.min()] = pts_middle1.min()
        if (pts_middle1.min() < pts_left.min()):
            pts_middle1[pts_middle1 < pts_left.min()] = pts_left.min()
        pts1 = np.hstack((pts_left, pts_middle1))
        # if (conf > 0.4) and (((tlx >= 10) and (tlx <= 313)) and ((tly <= 715)
        # and (tly >= 200))) and (((brx <= 615) and (brx >= 313)) and ((bry <=
        # 715) and (bry >= 200))):
        if (ret['middle1_lane_ymax'] > 7000):
            cv2.putText(
                undist,
                'Left lane is prohibited !',
                (50,
                 100),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0,
                 255,
                 255),
                1,
                lineType=cv2.LINE_AA)
            cv2.fillPoly(color_warp, np.int_([pts1]), (0, 0, 0))
        else:
            if (tlx_fst > 0) and (bry_fst >= 0) and (bry_fst <= 720):
                cv2.putText(
                    undist,
                    'Be careful ! There is a vehicle on the left lane.',
                    (50,
                     100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0,
                     255,
                     255),
                    1,
                    lineType=cv2.LINE_AA)
                cv2.fillPoly(
                    color_warp,
                    np.int_(
                        [pts1]),
                    ((-255 / 720) * bry_fst + 255,
                     (-255 / 720) * bry_fst + 255,
                        ((255 / 720) * bry_fst)))
                cv2.putText(
                    color_warp,
                    'Left',
                    (avg_left,
                     700),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.75,
                    (0,
                     0,
                     0),
                    5,
                    lineType=cv2.LINE_AA)
                if (int(pts_left.min()) >= 500) and (avg_fst > 75):
                    slow = cv2.addWeighted(color_warp[int(pts_left.min()):650, (avg_fst - 75):(
                        avg_fst + 75)], 1, slow[(int(pts_left.min()) - 500):150, 0:150], 1, 0)
                    color_warp[int(pts_left.min()):650,
                               (avg_fst - 75):(avg_fst + 75)] = slow

            else:
                if (tlx_fst == 0):
                    cv2.fillPoly(
                        color_warp,
                        np.int_(
                            [pts1]),
                        ((-255 / 720) * bry_fst + 255,
                         (-255 / 720) * bry_fst + 255,
                            ((255 / 720) * bry_fst)))
                    cv2.putText(
                        color_warp,
                        'Left',
                        (avg_left,
                         700),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.75,
                        (0,
                         0,
                         0),
                        5,
                        lineType=cv2.LINE_AA)
                else:
                    cv2.putText(
                        undist,
                        'Be careful ! There is a vehicle on the left lane.',
                        (50,
                         100),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.5,
                        (0,
                         255,
                         255),
                        1,
                        lineType=cv2.LINE_AA)
                    cv2.fillPoly(
                        color_warp,
                        np.int_(
                            [pts1]),
                        ((-255 / 720) * bry_fst + 255,
                         (-255 / 720) * bry_fst + 255,
                            ((255 / 720) * bry_fst)))
                    cv2.putText(
                        color_warp,
                        'Left',
                        (avg_left,
                         700),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.75,
                        (0,
                         0,
                         0),
                        5,
                        lineType=cv2.LINE_AA)
                    if (int(pts_left.min()) >= 500):
                        slow = cv2.addWeighted(color_warp[int(pts_left.min()):650, (avg_fst - 75):(
                            avg_fst + 75)], 1, slow[(int(pts_left.min()) - 500):150, 0:150], 1, 0)
                        color_warp[int(pts_left.min()):650,
                                   (avg_fst - 75):(avg_fst + 75)] = slow
                    else:
                        slow = cv2.addWeighted(
                            color_warp[500:650, (avg_fst - 75):(avg_fst + 75)], 1, slow[0:150, 0:150], 1, 0)
                        color_warp[500:650,
                                   (avg_fst - 75):(avg_fst + 75)] = slow

    else:
        if ((ret['middle1_lane_ymax'] > 7000)):
            cv2.putText(
                undist,
                'Left lane is prohibited !',
                (50,
                 100),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0,
                 255,
                 255),
                1,
                lineType=cv2.LINE_AA)
        else:

            if (tlx_fst > 0):
                cv2.putText(
                    undist,
                    'Be careful ! There is a vehicle on the left lane.',
                    (50,
                     100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0,
                     255,
                     255),
                    1,
                    lineType=cv2.LINE_AA)
            else:
                cv2.putText(
                    undist,
                    'Left lane is not accessible !',
                    (50,
                     100),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (0,
                     255,
                     255),
                    1,
                    lineType=cv2.LINE_AA)
 # --------------------------------------------------------------------------------

    if (middle1_fit != 'error') and (middle2_fit != 'error') and (
            ret['middle1_lane_ymax'] > 100) and (ret['middle2_lane_ymax'] > 100):
        slow = cv2.imread('media/signs/Webp.net-resizeimage.png')
        pts_middle1 = np.array(
            [np.flipud(np.transpose(np.vstack([middle1_fitx, ploty_middle])))])
        pts_middle2 = np.array(
            [np.transpose(np.vstack([middle2_fitx, ploty_middle]))])
        avg_middle1 = int(sum(middle1_fitx) / len(middle1_fitx))
        avg_middle2 = int(sum(middle1_fitx) / len(middle1_fitx))
        avg_middle = int((avg_middle1 + avg_middle2) / 2)

        if (pts_middle1.min() < pts_middle2.min()):
            pts_middle1[pts_middle1 < pts_middle2.min()] = pts_middle2.min()
        if (pts_middle2.min() < pts_middle1.min()):
            pts_middle2[pts_middle2 < pts_middle1.min()] = pts_middle1.min()
        pts2 = np.hstack((pts_middle1, pts_middle2))
        if (tlx_scnd > 0):
            # if ((ret['middle1_lane_nonzerox'] +
            # ret['middle2_lane_nonzerox']) > 100):
            cv2.putText(
                undist,
                'There is a vehicle infront of you !',
                (50,
                 50),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0,
                 255,
                 255),
                1,
                lineType=cv2.LINE_AA)
            cv2.fillPoly(
                color_warp,
                np.int_(
                    [pts2]),
                (0,
                 (-255 / 720) * bry_scnd + 255,
                    (255 / 720) * bry_scnd))
            cv2.putText(
                color_warp,
                'Main',
                (avg_middle + 50,
                 700),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.75,
                (0,
                 0,
                 0),
                5,
                lineType=cv2.LINE_AA)
            if (int(pts_middle1.min()) >= 500):
                slow = cv2.addWeighted(color_warp[int(pts_middle1.min()):650, (avg_middle + 50):(
                    avg_middle + 200)], 1, slow[(int(pts_middle1.min()) - 500):150, 0:150], 1, 0)
                color_warp[int(pts_middle1.min()):650,
                           (avg_middle + 50):(avg_middle + 200)] = slow
            else:
                slow = cv2.addWeighted(
                    color_warp[500:650, (avg_middle + 50):(avg_middle + 200)], 1, slow[0:150, 0:150], 1, 0)
                color_warp[500:650, (avg_middle + 50)                           :(avg_middle + 200)] = slow
        else:
            cv2.fillPoly(
                color_warp,
                np.int_(
                    [pts2]),
                (0,
                 (-255 / 720) * bry_scnd + 255,
                    (255 / 720) * bry_scnd))
            cv2.putText(
                color_warp,
                'Main',
                (avg_middle + 50,
                 700),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.75,
                (0,
                 0,
                 0),
                5,
                lineType=cv2.LINE_AA)
    else:
        cv2.putText(
            undist,
            'Cannot detect the current lane!',
            (50,
             50),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0,
             255,
             255),
            1,
            lineType=cv2.LINE_AA)

    return color_warp
