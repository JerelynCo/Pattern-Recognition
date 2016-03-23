# Pattern-Recognition Lab 2
By Jerelyn Co and Hadrian Lim

## Requirements
Python 3 - Anaconda Distribution
OpenCV2 

## Files and Directories
* extract_ROI.py
	- extracts all ROI via contour detection inside the sample_pics directory, applies Sobel filtering, and outputs CSV files of the features per image.
	- Directories processed and ROI contains image outputs together with the CSV files
* classifier/classifier.py
	- grabs a CSV file, normalized_all.csv, which assumes a manually classified dataset from the sample_images directory
	- separates dataset into training and testing sets
	- Applies Naive Baysian and outputs the parameters in a JSON file, train_summary.json
* predict_objects.py
	- Uses train_summary.json and classifies images in test_pics (create your own dir and upload your own photos)
* webcam_prediction.py
	- a webcam implementation of the predict_objects.py
* saved/
	- contains a backup of manually classified ROI from images

## Answers to Questions:
1) How would you determine the accuracy of your Naive Bayes model in this
case? Use a confusion matrix as your final result but also explain your
methodology

 ___________________________
 |_________Predicted________|
 |_____|___Yes__|___No______|	
R|Yes__|    3   |     1     |
E|_____|________|___________|
A|No___|    12  |     19    |
L|_____|________|___________|

Our methodology to calculate the accuracy of the model was that we got the sensitivity (true positive rate) and specificity (true negative rate) from a test_dataset and then getting the accuracy, which is the sum of the sensitivity and specificity, divided by the total population.
From our confusion matrix, we calculated that our model accuracy is 62.852%

	Calculation: ACC = TP + TN / TOTAL_POP
				 TP = 3, TN = 19, TOTAL_POP = 35

2) Will Naive Bayes work for this case? What improvements might you suggest to improve this classification system?

Naive Bayes works best if and only if the provided training set is of good quality, meaning that it bases its classification on the provided training set. Furthermore, the number of instances per label must be  equally distributed or representative of a population because the algorithm relies on this for classification. Some improvements that maybe introduced in this system is to use ensemble or boosting methods to tweak the system, or even creating a network of naive bayes classifiers for different kinds of objects.