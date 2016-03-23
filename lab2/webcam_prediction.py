import numpy as np
import cv2
import math
import json


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


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = auto_canny(blurred)

    contoured_image, contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ddepth for sobel
    ddepth = cv2.CV_8U

    sobel_x = cv2.Sobel(blurred, ddepth, 1, 0)
    sobel_y = cv2.Sobel(blurred, ddepth, 1, 0)
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    counter = 0
    image_out = frame.copy()

    for contour in contours:
        # get box bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        if h < image_out.shape[1] * 0.05 or w < image_out.shape[0] * 0.05:
            continue

        crop_img_sobel = sobel[y:y + h, x:x + w]

        resized = cv2.resize(crop_img_sobel, (8, 8)).flatten() / 255

        with open('classifier/train_summary.json', 'r') as f:
            train_summary = json.loads(f.read())
        if(predict(train_summary, resized) == '1'):
            cv2.rectangle(edges, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            cv2.rectangle(edges, (x, y), (x + w, y + h), (0, 0, 255), 2)

        counter += 1

    # Display the resulting frame
    cv2.imshow('frame', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
