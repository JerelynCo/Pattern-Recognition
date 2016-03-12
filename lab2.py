import cv2
import numpy as np
import pandas as pd


# cd 'C:\\Users\\hadri\\Google Drive\\Acads\\CS 129.18\\OpenCV'

def showImage(winName, image):
    cv2.imshow(winName, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# From
# http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

image = cv2.imread("balloons.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Common use case of 5x5
edges = auto_canny(blurred)
showImage('Canny', edges)

contoured_image, contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ROI -> Regions of Interest; bb: bounding box
ROI_bb = pd.DataFrame(columns=['y', 'y1', 'x', 'x1'])
# Will contain 8x8 feature vectors for BING
ROI_features = pd.DataFrame(columns=['feature_' + str(x) for x in range(64)])
counter = 0
image_out = image.copy()

# ddepth for sobel
ddepth = cv2.CV_8U

# Draw bounding boxes enclosing contours in orig image
for contour in contours:
    # get box bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # discard areas that are too large
    # if h > image.shape[1] * 0.7 and w > image.shape[0] * 0.7:
    #     continue

    # discard areas that are too small (5% of the size of image)
    if h < image.shape[1] * 0.05 or w < image.shape[0] * 0.05:
        continue

    # crop ROI, store bounding box info, store img mat, save img
    crop_img = image[y:y + h, x:x + w]
    crop_img_gray = gray[y:y + h, x:x + w]
    ROI_bb.loc[counter] = [y, y + h, x, x + h]
    cv2.imwrite('ROI/' + str(counter) + '.jpg', crop_img)

    # Doing BING -> http://mmcheng.net/bing/
    # Do Sobel for x and y (set dx=1, dy=0, viceversa)
    sobel_x = cv2.Sobel(crop_img_gray, ddepth, 1, 0)
    sobel_y = cv2.Sobel(crop_img_gray, ddepth, 0, 1)
    # merge sobel dx and dy images via cv2.addWeight with 0.5 alpha/beta,
    # 0 gamma
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    # Resize to 8x8 image, flatten, and store to ROI_features
    sobel_flat = cv2.resize(sobel, (8, 8)).flatten()
    ROI_features.loc[counter] = sobel_flat
    # draw a box around contour on original image
    cv2.rectangle(image_out, (x, y), (x + w, y + h), (255, 0, 255), 2)
    counter += 1

showImage('Bounded Rectangles', image_out)
ROI_bb.to_csv('ROI_coords.csv', index=False)
ROI_features.to_csv('ROI_features.csv', index=False)
# have not implemented 1 e and f though i think not needed
# C++ specifics kasi

# Naive Bayes to follow
