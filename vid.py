import numpy as np
import cv2


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

    counter = 0
    image_out = frame.copy()

    for contour in contours:
        # get box bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        if h < image_out.shape[1] * 0.05 or w < image_out.shape[0] * 0.05:
            continue
        # draw a box around contour on original image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        counter += 1

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
