import cv2
import numpy as np
import os
import json
import math

def showImage(winName, image):
    cv2.imshow(winName, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# From:
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


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def ROI(image, gray, edges):
    contoured_image, contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        # discard areas that are too small (8% of the size of image)
        if h < image.shape[1] * 0.08 or w < image.shape[0] * 0.08:
            continue

        crop_img_gray = gray[y:y + h, x:x + w]

        # Doing BING -> http://mmcheng.net/bing/
        # Do Sobel for x and y (set dx=1, dy=0, viceversa)
        sobel_x = cv2.Sobel(crop_img_gray, ddepth, 1, 0)
        sobel_y = cv2.Sobel(crop_img_gray, ddepth, 0, 1)
        # merge sobel dx and dy images via cv2.addWeight with 0.5 alpha/beta,
        # 0 gamma
        sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
        # Resize to 8x8 image, flatten, and store to ROI_features
        sobel_flat = cv2.resize(sobel, (8, 8)).flatten()/255

        with open('classifier/train_summary.json', 'r') as f:
            train_summary = json.loads(f.read())
        if(predict(train_summary, sobel_flat) == '1'):
            cv2.rectangle(image_out, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(image_out, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # ROI_features.loc[counter] = sobel_flat
        # draw a box around contour on original image

    cv2.imwrite("nb_" + 'lol.jpeg', image_out)


def main():

    # directories
    pictures_dir = "sample_pics/"
    ROI_dir = "ROI/"
    processed_dir = "processed/"

    if not os.path.exists(ROI_dir):
        os.makedirs(ROI_dir)

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    img_fn = "/home/jerelynco/Documents/Acads/2 Sem/CS129.18/Pattern-Recognition/lab2/test_pics/webcam1.jpg"
    image = cv2.imread(img_fn)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Common use case of 5x5
    edges = auto_canny(blurred)
    ROI(image, gray, edges)

if __name__ == '__main__':
    main()
