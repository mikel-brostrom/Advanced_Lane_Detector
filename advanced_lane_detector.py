import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box, Point


def lane_poly_fit(img, img_binary, birdeye_binary, verbose):

    n_win = 8
    h, w = birdeye_binary.shape[0], birdeye_binary.shape[1]
    # Take a histogram of the bottom half of the image
    histogram = np.sum(birdeye_binary[h // 2:-30, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255

    # Set height of windows
    window_height = np.int(h / n_win)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = np.where(birdeye_binary > 0)

    points = [Point(x, y) for x, y in zip(nonzero[1], nonzero[0])]

    margin = 100  # width of the windows +/- margin

    left_lane_boxes_points = []
    right_lane_boxes_points = []

    # Step through the windows one by one
    for window in range(n_win):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = h - (window + 1) * h
        win_y_high = h - window * window_height
        win_xleft_low = leftx_base - margin
        win_xleft_high = leftx_base + margin
        win_xright_low = rightx_base - margin
        win_xright_high = rightx_base + margin

        # draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # create box classes
        left_box = box(win_xleft_low, win_y_low, win_xleft_high, win_y_high)
        right_box = box(win_xright_low, win_y_low, win_xright_high, win_y_high)

        # check if the current left box contains any of the lane points
        left_box_points = [l for l in points if left_box.contains(l)]
        # calculate the x and y mean of all the points associated to a box if it
        # is not empty
        if len(left_box_points) != 0:
            mean_left_box_points_x = np.mean([point.x for point in left_box_points])
            mean_left_box_points_y = np.mean([point.y for point in left_box_points])
            left_lane_boxes_points.append(Point(mean_left_box_points_x, mean_left_box_points_y))

        # check if the current right box contains any of the lane points
        right_box_points = [l for l in points if right_box.contains(l)]
        # calculate the x and y mean of all the points associated to a box if it
        # is not empty
        if len(right_box_points) != 0:
            mean_right_box_points_x = np.mean([point.x for point in right_box_points])
            mean_right_box_points_y = np.mean([point.y for point in right_box_points])
            right_lane_boxes_points.append(Point(mean_right_box_points_x, mean_right_box_points_y))

        #print(len(left_lane_boxes_points), len(right_lane_boxes_points))

    # fit a second degree polynom to averaged point generated in each non-empty box
    left_lane_points = np.polyfit([int(point.y) for point in left_lane_boxes_points],
                                  [int(point.x) for point in left_lane_boxes_points], 2)
    right_lane_points = np.polyfit([int(point.y) for point in right_lane_boxes_points],
                                   [int(point.x) for point in right_lane_boxes_points], 2)

    # create equidistant points from 0 to 720
    y = np.linspace(0, h - 1, h)
    # sample the function
    left_fit = left_lane_points[0] * y ** 2 + left_lane_points[1] * y + left_lane_points[2]
    right_fit = right_lane_points[0] * y ** 2 + right_lane_points[1] * y + right_lane_points[2]

    # Plot points corresponding to right and left lane
    for point in left_lane_boxes_points:
        cv2.circle(out_img, (int(point.x), int(point.y)),
                   radius=5,
                   color=[0, 0, 255],
                   thickness=10)

    for point in right_lane_boxes_points:
        cv2.circle(out_img, (int(point.x), int(point.y)),
                   radius=5,
                   color=[0, 0, 255],
                   thickness=10)

    if verbose:
        print('There are:', len(points), 'non-empty points')
        f, axarray = plt.subplots(5, 1)
        cv2.circle(img, (w // 2 + 75, 460),
                   radius=25,
                   color=[0, 0, 255],
                   thickness=10)
        cv2.circle(img, (w // 2 - 75, 460),
                   radius=25,
                   color=[0, 0, 255],
                   thickness=10)
        axarray[0].imshow(cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB))
        axarray[1].imshow(img_binary, cmap='gray')
        axarray[2].imshow(birdeye_binary, cmap='gray')
        axarray[3].imshow(cv2.cvtColor(out_img, code=cv2.COLOR_BGR2RGB))
        axarray[4].plot(left_fit, y, color='yellow')
        axarray[4].plot(right_fit, y, color='yellow')
        axarray[4].set_xlim(0, 1280)
        axarray[4].set_ylim(720, 0)
    return left_fit, right_fit


def birdeye(img, verbose):
    h, w = img.shape[0], img.shape[1]

    src = np.float32([[w, h - 20],    # bottom right
                      [0, h - 20],    # bottom left
                      [w // 2 - 90, 460],   # top left
                      [w // 2 + 90, 460]])  # top right
    dst = np.float32([[w, h],       # bottom right
                      [0, h],       # bottom left
                      [0, 0],       # top left
                      [w, 0]])      # top right

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    img_birdeye = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    if verbose:
        f, axarray = plt.subplots(1, 2)
        axarray[0].imshow(img, cmap='gray')
        axarray[1].imshow(img_birdeye, cmap='gray')

    return img_birdeye, M, Minv


def binarize(frame, verbose):
    h, w = frame.shape[0], frame.shape[1]

    # create a placeholder for the final mask
    mask = np.zeros(shape=(h, w), dtype=np.uint8)

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # augment contrast
    eq_hist = cv2.equalizeHist(gray)
    # threshold it to get a mask
    _, eq_hist_mask = cv2.threshold(eq_hist, thresh=251, maxval=255, type=cv2.THRESH_BINARY)
    binary = np.logical_or(mask, eq_hist_mask)

    # apply light morphology to remove binary noise
    kernel = np.ones((5, 5), np.uint8)
    filtered_mask = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if verbose:
        plt.imshow(filtered_mask, cmap='gray')

    return filtered_mask


def draw_lane(img, img_binary, img_birdeye, Minv, left_fit, right_fit, verbose):
    h, w = img_birdeye.shape[0], img_birdeye.shape[1]
    y = np.linspace(0, h - 1, h)

    # draw road as green polygon on original frame
    road_warp = np.zeros_like(img, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fit, y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit, y])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))

    road_dewarped = cv2.warpPerspective(road_warp, Minv, (w, h))
    blend_onto_road = cv2.addWeighted(src1=img,
                                      alpha=0.8,
                                      src2=road_dewarped,
                                      beta=0.5,
                                      gamma=0.)
    if verbose:
        plt.imshow(cv2.cvtColor(blend_onto_road, code=cv2.COLOR_BGR2RGB))

    return blend_onto_road


def process_pipeline(frame, show_result_img):
    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binary = binarize(frame, verbose=False)

    # compute perspective transform to obtain bird's eye view
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    # calculate fitted polynoms in transformed image
    left_fit, right_fit = lane_poly_fit(frame, img_binary, img_birdeye, verbose=False)

    # draw the surface enclosed by lane lines back onto the original frame
    blend_onto_road = draw_lane(frame,
                                img_binary,
                                img_birdeye,
                                Minv,
                                left_fit,
                                right_fit,
                                verbose=show_result_img)

    return blend_onto_road


if __name__ == '__main__':

    verbose = True

    test_img_dir = 'test_images'
    for test_img in os.listdir(test_img_dir):
        print(test_img)

        frame = cv2.imread(os.path.join(test_img_dir, test_img))

        img_result = process_pipeline(frame, show_result_img=True)

        cv2.imwrite('output_images/{}'.format(test_img), img_result)

        if verbose:
            plt.waitforbuttonpress()
            plt.close()
