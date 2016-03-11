import cv2
import numpy as np


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
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edges = auto_canny(blurred)
showImage('Canny', edges)

contoured_image, contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles enclosing contours in orig image
for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h > image.shape[1] * 0.7 and w > image.shape[0] * 0.7:
        continue

    # discard areas that are too small
    if h < image.shape[1] * 0.05 or w < image.shape[0] * 0.05:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

showImage('Bounded Rectangles', image)

# have not implemented 1 e and f though i think not needed

