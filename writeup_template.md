# **Traffic Sign Recognition** 

## Writeup


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

[image1]: ./Input_data_visualization.jpg "Visualization"
[image2]: ./Random_input_images.jpg "Visualization"
[image3]: ./Grayscale.jpg "Grayscaling"
[image4]: ./Random_rotation.jpg "Random Rotation"
[image5]: ./test_images/images-1.jpeg "Traffic Sign 1"
[image6]: ./test_images/images-2.jpeg "Traffic Sign 2"
[image7]: ./test_images/images-3.jpeg "Traffic Sign 3"
[image8]: ./test_images/images-4.jpeg "Traffic Sign 4"
[image9]: ./test_images/images-5.jpeg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

Once I read the training, validation and test data sets, I used pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. Below are some of the traffic signs and their corresponding labels. 

![alt text][image2]

Now let's look at a bar chart showing traffic signs and their counts in the training data. We can see that some of the traffic signs don't have enough images in our training data.

![alt text][image1]

### Design and Test a Model Architecture

#### Preprocessing and Augmenting Training Set:

As a first step, I decided to convert the images to grayscale because training my network on grayscale images gave me a higher classification accuracy. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data to (0,1) since I achived better accuracy with (0,1) normalizantion than (-1,1).

I also decided to generate additional data because some of the labels didn't have enough images in the training data. 

To add more data to the the data set, I used generated more copies of images for labels with less than 800 images and I added a random rotation to the new images.

Here is an example of an original image and an augmented image:

![alt text][image4]

The difference between the original data set and the augmented data set is the the size of training set increased from 34799 to 43880.


#### Final Model Architecture:

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:--------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					
|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					
|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400	|
| RELU					|												|
| Dropout   	      	| 0.5 keep_prob 								
|
| Flatten   	      	| outputs 400   								
|
| Fully connected		| Input 400, outputs 120						|
| RELU					|												|
| Dropout   	      	| 0.5 keep_prob 								
|
| Fully connected		| Input 120, outputs 43 						|
 

#### Training the Model:

To train the model, I used adam optimizer with 0.0009 learning rate. I also used a batch size of 128 and 60 epochs to train my model.

I initially used a learnign rate of 0.001 which gave me an accuray of 94% on my validation set. Then I decided to reduce my learning rate and increase the number of epochs which gave a validation accuracy of 96%.

My final model results were:
* validation set accuracy of 96.3%
* test set accuracy of 95%

I first used the LeNet architecture to train and got an accuracy of 87.6%. And so, I changed my network architecture from the LeNet architecture to increase number of convolution layers, decrease the number of fully connected layers and include dropouts which helped increase my accuracy to 96.3%. The reason I chose to increase convolution layers is the deeper the model is, the better it understands the images. At the same time, I wanted to avoid overfitting of the model by introducing dropouts.

### Test of the Model on New Images:

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The third image was classified wrong because it wasn't present in the training data set.

Here are the results of the prediction:

| Image			        				|     Prediction	        	| 
|:-------------------------------------:|:-----------------------------:| 
| Stop Sign      						| Stop sign 					| 
| No Entry     							| No Entry 						|
| Right-of-way at the next intersection	| Right-of-way at the next intersection|
| Road Work 					      	| Road Work 					 |
| No Right Turn 						| Speed limit (30km/h) 			|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95%.

#### 3. How certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction:


The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the fourth image, the model is relatively sure that this is a no entry sign (probability of 0.59), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .030         			| Stop sign   									| 
| .42     				| Right-of-way at the next intersection 		|
| .59					| No entry 										|
| .13	      			| Road Work 					 				|
| .15				    | No Right Turn      							|

For the third image the prediction is wrong since that image is not in the training set (No Right Turn).

