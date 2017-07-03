# **Traffic Sign Recognition** 

## Writeup Template

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


<!--
[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
-->

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ompugao/CarND-Traffic-Sign-Classifier-Project/blob/shohei/Traffic_Sign_Classifier.ipynb)
And, I bundled the exported html file. here is [the link to it](https://github.com/ompugao/CarND-Traffic-Sign-Classifier-Project/blob/shohei/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

here is the summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![sample_input][materials/sampel_input.jpg]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 

As a first step, I decided to convert the images to grayscale.

As a second step, I normalized the image data.
I got a better result by `X - np.mean(X)` rather than `(X - np.mean(X))/256` somehow (probably because the differentiation becomes small?).

As a third step, when I looked through the input data, I found images which do not have a flat brightness.
![00001_00029.ppm]('materials/00001_00029.ppm')
So, I modified the model to use local response normalization technique to normalize the brightness/contrast of images.
This change also brought me a better result.

As a fourth step, I decided to augment dataset because I could not expect a much-better result by finding a better normalization.
I applied random-rotation(0~20[deg]) and width/height shifting(~0.1*(widht|height) pixels shift).
I didn't apply flipping, too-much rotation, and affine-transformation(distortion) because traffic signs should not have such kind of variation.
This data augmentation brought me a huge improvement and I got 0.940 accuracy.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers, LeNet+lrn:

| Layer         				|     Description	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Input       	  				| 32x32x1 Gray image   							| 
| Convolution 5x5     			| 1x1 stride, valid padding, outputs 28x28x6 	|
| Local Response Normalization	| depth 4, bias 1, alpha 0.001/9, beta 0.75		|
| RELU							|												|
| Max pooling	      			| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 5x5	    		| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU							|												|
| Max pooling	      			| 2x2 stride,  outputs 5x5x16	 				|
| Flatten						| outputs 400									|
| Fully connected				| outputs 120  									|
| RELU							|												|
| Fully connected				| outputs 84  									|
| RELU							|												|
| Fully connected				| outputs 10  									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I use the following parameters:
rate = 0.001
EPOCHS = 40
BATCH_SIZE = 128

I increased EPOCHS because I found that the default number of epoch was not enough for the optimization to converge,
but I didn't change other parameters.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.950
* test set accuracy of 0.940

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    * LeNet, because there was the sample code (I should check other networks, like Alexnet, but I thought this would be a good staring point).
* What were some problems with the initial architecture?
    * no normalizations and no data augmentations.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    * I added a local response normalization layer after the first convolution layer, because when I looked through the input data, I found images which do not have a flat brightness.
    * (I'm not still sure the good practice to change and adjust models. There are plenty amount of options to take and every option seems to work intuitively. how do people pick up one out of options? like 'ok, I guess I should insert a new layer between these layers with these parameters.'?)
* Which parameters were tuned? How were they adjusted and why?
    * I tuned only the number of epochs because it was not enough for the optimization to converge when I was checking the accuracy increase.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * there are only two convolution layers at the very begining of the network, so I guess the current model will only extract and have tiny local features (about 25x25 pixels?). I can insert more convolution layers into the middle area of the model so that it can have global charasteristic features.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![20kmph][test_images/20kmph.jpg] 
![children_crossing][test_images/children_crossing.jpg] 
![no_entry][test_images/no_entry.jpg] 
![right_turn [test_images/right_turn.jpg] 
![stop][test_images/stop.jpg] 

borrowed from https://github.com/netaz/carnd_traffic_sign_classifier, originally from http://electronicimaging.spiedigitallibrary.org/data/journals/electim/927109/jei_22_4_041105_f010.png

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 20kmph	      		| Road Work   									| 
| children crossing 	| Dangerous curve to the left					|
| No Entry				| No Entry										|
| Turn Right      		| Turn Right 					 				|
| Stop					| Stop      									|

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.
It seems pretty bad, but I cannot say it is bad because the number of (new) samples is small.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a stop sign, but it is wrong.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 25,Road work   								| 
| .00     				| 37,Go straight or left						|
| .00					| 30,Beware of ice/snow							|
| .00	      			| 18,General Caution				 			|
| .00				    | 11,Right-of-way at the next intersection		|

For the second image, the model is at a loss among bicycles/road work/pedestrians, but it is children.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .59         			| 29,Bicycles crossing							| 
| .20     				| 25,Road work									|
| .10					| 27,Pedestrians	 							|
| .00	      			| 28,Children crossing				 			|
| .00				    | 11,Right-of-way at the next intersection		|

In the following three cases, the model returns correct answers and  the results are pretty straightforward to understand.

For the third image, the model is pretty sure that this is a 'no entry' sign and it's correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .98         			| 17,No entry									| 
| .01     				| 13,Yield										|
| .00					| 39,Keep left	 								|
| .00	      			| 9,No passing				 					|
| .00				    | 32,End of all speed and passing limits		|

For the fourth image, the model is pretty sure that this is a 'turn right' sign and it's correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 33,Turn right ahead							| 
| .00     				| 36,Go straight or right						|
| .00					| 35,Ahead only	 								|
| .00	      			| 11,Right-of-way at the next intersection		|
| .00				    | 37,Go straight or left						|

For the fifth image, the model is pretty sure that this is a 'stop' sign and it's correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .85         			| 14,Stop										| 
| .14     				| 4,Speed limit (70km/h)						|
| .00					| 2,Speed limit (50km/h)	 					|
| .00	      			| 1,Speed limit (30km/h)		 				|
| .00				    | 6,End of speed limit (80km/h)					|

