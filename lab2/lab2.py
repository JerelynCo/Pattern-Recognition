import cv2
import numpy as np
import pandas as pd
import os

def showImage(winName, image):
    cv2.imshow(winName, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# From: http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def ROI(edges):
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

	    # discard areas that are too small (8% of the size of image)
	    if h < image.shape[1] * 0.08 or w < image.shape[0] * 0.08:
	        continue

	    # crop ROI, store bounding box info, store img mat, save img
	    crop_img = image[y:y + h, x:x + w]
	    crop_img_gray = gray[y:y + h, x:x + w]
	    ROI_bb.loc[counter] = [y, y + h, x, x + h]
	    cv2.imwrite(ROI_dir + ROI_subdir + str(counter) + '.jpg', crop_img)

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

	cv2.imwrite(processed_dir + "bounded_" + img_fn, image_out)
	ROI_bb.to_csv(ROI_dir + ROI_subdir + ROI_coords_fn, index=False)
	ROI_features.to_csv(ROI_dir + ROI_subdir + ROI_features_fn, index=False)
	return ROI_features

## filenames
ROI_coords_fn = "ROI_coords.csv"
ROI_features_fn = "ROI_features.csv"

## directories
pictures_dir = "sample_pics/"
ROI_dir = "ROI/"
processed_dir = "processed/"

if not os.path.exists(ROI_dir):
	os.makedirs(ROI_dir)

if not os.path.exists(processed_dir):
	os.makedirs(processed_dir);

for file in os.listdir(pictures_dir):
	global img_fn
	global ROI_subdir
	img_fn = file
	ROI_subdir = img_fn.split(".")[0] + "/"

	if not os.path.exists(ROI_dir + ROI_subdir):
		os.makedirs(ROI_dir + ROI_subdir)
	
	image = cv2.imread(pictures_dir + img_fn)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Common use case of 5x5
	edges = auto_canny(blurred)
	cv2.imwrite(processed_dir + "canny_" + img_fn, edges)
	roi = ROI(edges)
