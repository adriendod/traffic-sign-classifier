# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/1.PNG "Histogram"
[image2]: ./images/2.PNG "New Images"
[image3]: ./images/3.PNG "Features"


## Rubric Points
 

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 10

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step. I tried the model with grayscale image and with color image.
I decided not to convert the images to grayscale because the resolution is already very small. Color will help differenciate some of the sign.

I only normalized the image dividing the pixel values by 255.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Flatten		| output : 1600        									|
| Fully Connected. | Output = 300		|	     									|
| RELU					|		
| Dropout					|		keep_prob = 0.5
| Fully Connected. | Output = 200			     									|
| RELU					|
| Dropout					|		keep_prob = 0.5   |
| Fully Connected. | Output = 75			     									|
| Softmax| Output = 10|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model with and Adam Optimizer, 30 epochs (best model was saved), a 64 batch size and a learning_rate of 0.001 (lower or higher doesnt work well)

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training Accuracy = 1.000
* Validation Accuracy = 0.979
* Test Accuracy = 0.954

If an iterative approach was chosen:
First i ran the lenet model and reached a validation accurasy around 0.89. I started to add another FC and made them deeper. The accuracy improved a bit.
Then I added dropout and i reached over the goal of 0.93.
I then decided to use a lot more filters in the convolution layers to reach a 0.979 validation accuracy.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] 

The second image and the 3rd might be difficult to classify because the sign is pretty small in the image.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection     		| Right-of-way at the next intersection 									| 
| Stop Sign  			| Priority road										|
| Speed limit (30km/h)				| Speed limit (50km/h)										|
| Road work      		| Road work  				 				|
| Children crossing			| Children crossing	     							|
| Pedestrians			| General caution    							|
| Stop Sign		| Stop Sign     							|


The model was able to correctly guess 4 of the 7 traffic signs, which gives an accuracy of 57%. This is not great but in re

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a Right-of-way at the next intersection   (probability of 0.64), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .64         			| Right-of-way at the next intersection     									| 
| .18     				| Double curve										|
| .14					| Priority road										|
| .14	      			| Roundabout mandatory			 				|
| .11				    | Beware of ice/snow    							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

We can see the shape of STOP later are highlighted. The shape of the sign is also used.

![alt text][image3] 
