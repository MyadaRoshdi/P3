# **Behavioral Cloning** 

## Writeup 

### This file is my writeup report that covers my work for the [Behavioral Cloning Project](https://github.com/MyadaRoshdi/P3).
1st step to use my project, download it locally:
  - git clone https://github.com/MyadaRoshdi/P3
  - cd P3

2nd step, get the dataset.
  - My dataset is collected manually from that given [simulator](https://github.com/udacity/self-driving-car-sim), and then augmented with the [Sample training set](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) offered by udacity. More details about how I collected my data is described below.
  
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-Nvidia-architecture.png "Model Visualization"
[image2]: ./examples/Overfitting.png "Model Overfitting"
[image3]: ./examples/No-Overfitting.png "Model without Overfitting"
[image4]: ./examples/forward_center.jpg "Center Camera Image"
[image5]: ./examples/forward_left.jpg "Left Camera Image"
[image6]: ./examples/forward_right.jpg "right Camera Image"
[image7]: ./examples/placeholder_small.png "un-touched Image"
[image8]: ./examples/placeholder_small.png "Cropped Image"
[image9]: ./examples/placeholder_small.png "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. I also organized this report in a form of Questions & Answers which Cover all rubric points. 

---
### 1. Files Submitted & Code Quality

#### 1.1. Submission includes all required files and can be used to run the simulator in autonomous mode
**Question1: Are all required files submitted?**

My project includes the following files:
* model.py: containing the script to create and train the model
* drive.py: a script for driving the car in autonomous mode
* model.h5: containing a trained convolution neural network 
* writeup_report.md: summarizing the results
* Readme.md giving a high level description of the project
* examples: this folder contains all images used in the writeup_report and Readme files.

#### 1.2. Submission includes functional code
**Question2: Is the code functional?**

Using the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim) and my [drive.py](https://github.com/MyadaRoshdi/P3/blob/master/drive.py) file feeding it with my trained model [model.h5]() the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 1.3. Submission code is usable and readable
**Question3: Is the code usable and readable?**

The model.py file contains the code for training and saving the convolution neural network, here I used the Nvidia model as shown [here] (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) . The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### 2. Model Architecture and Training Strategy

#### 2.1. An appropriate model architecture has been employed
**Question4: Has an appropriate model architecture been employed for the task?**

My model Uses deep convolutional neural network that takes images captured by the 3-cameras in the simulator and returns the steering angles. I used the Nvidia architecture as shown below:

![alt text][image1]

 here's a detailed description of the model layers I used:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 BGR image           | 
| cropping          | Output 80x320x3               |
| Lambda            | Normalization                  |
| Convolution 5x5   | 2x2 stride, depth 24,regularization l2(0.01)  	|
| RELU				    	|	Activation											|
| Convolution 5x5   | 2x2 stride, depth 36, regularization l2(0.01) 	|
| RELU				    	|	Activation											|		
| Convolution 5x5   | 2x2 stride, depth 48, regularization l2(0.01) 	|
| RELU				    	|	Activation											|
| Convolution 3x3   |  depth 64, regularization l2(0.01) 	|
| RELU				    	|	Activation											|
| Convolution 3x3   |  depth 64, regularization l2(0.01) 	|
| RELU				    	|	Activation											|
| Flatten   	  	| Flatten  o/p of last conv layer	|				
| Fully connected		| Dense,  Output = 100    |
| Fully connected		| Dense,  Output = 50    |
| Fully connected		| Dense,  Output = 10    |
| Fully connected		| Dense,  Output = 1    |
|						|												|
|						|												|
 


I used 5- convolutional layers (model.py lines 18-24)  followed a flatten layer, then 4- fully connected layers 

My model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2.2. Attempts to reduce overfitting in the model
**Question5: Has an attempt been made to reduce overfitting of the model?**

Yes, The model contains L2 regulariztion after every convolutional layer in order to reduce overfitting (model.py lines 21). 

Here is the model mean square error(MSE) graph, showing how MSE in validation is much bigger than MSE in training, which means data overfitting:

![alt text][image2]

However, you can see in the following graph, how applying L2-regularization enhanced data overfitting

![alt text][image3]

#### 2.3. Model parameter tuning
**Question6: Have the model parameters been tuned appropriately?**

My model used stochastic optimization (Adam optimizer) to reduce the mean squared error on steering angle, so the learning rate was not tuned manually (model.py line 25).

To train the model, I used the following values:

* Type of optimizer: AdamOptimizer

* number of epochs: 3

* steering correction : 0.2 left and 0.2 right

* batch_size = 32

**NOTE**  All parameters data has been chosen empirically.


#### 2.4. Appropriate training data
**Question7: Is the training data chosen appropriately?**

Training data was chosen to keep the vehicle driving on the road. My data was a combination of the following:

1) Sample training data supported by project

2) 3- full tracks of center lane driving
 Here is an example image of forward middle of the lane driving views from Left, center and right Cameras :

![Center_Camera_View][image4] Centeral Camera
![Left_Camera_View][image5] Left Camera 
![Right_Camera_View][image6] Right Camera

3) Recovering from the left and right sides

4) 1- counterclock wise track

I also combined images from  central Camera, left and right Camera with steering angle correction = 0.2. 

To augment the data , I also flipped images and angles from central camera. For example, here is an image that has then been cropped and flipped :

![Original_Untouched_Image][image7]

![Cropped_Image][image8]

![flipped_Image][image9]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

Overall, my datasets sizes are:

- Training dataset size = 39,572 images

- Validation dataset size = 9,896 images


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as when I increase, the loss starts to increase, so I stopped by 3-epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.


**NOTE** Collecting data in this project is really very challenging, specially on my 64-bit windows and using keys, I would suggest to use video games handhelds for more controlling and smooth driving.



### 3. Model Architecture and Training Strategy

#### 3.1. Solution Design Approach
**Question8: Is the solution design documented?**

The overall strategy for deriving a model architecture was as following:

a) I started first only with  the suggested [Sample training dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

b) My first step was to use a model similar to the  [Modified lenet](https://github.com/MyadaRoshdi/P2/blob/master/LeNet5_Models/2-stage-ConvNet-architecture.png) architecture. I adjusted the i/p and o/p dimensions to use it, I thought to use this model as it was very successful in the traffic signs classifier I built, and it already contains a good number of convolutional layers and fully connected ones. but my car was not doing the expected behavior on the simulator even when I tested on another collected dataset from the simulator.

c) Then I moved to the Nvidia model, which really did a big improvment in the car behavior, but when I test my car autonomously in the simulator for this trained model and data, it used to stop when the car reaches the bridge.

d) I started to collect more data and augment with the Sample training dataset. I Also used the hint that "Keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py .", so I preprocessed my images before feeding to the model training, by smooting them then convert to YUV colorspace, and actually I had a big improvement after applying that.

e) In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used the rule of thump of spliting 20% of the training set to validation set. I found that the model and data I had gives a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

f) To combat the overfitting, I modified the model so that I added L2-regularization, there's no overfitting now anymore. look at section 2.2., for visualization. 

g) Now, testing my car on the simulator, then it starts to get out of the road near the end of the track. Then I re-collected some data for good driving at this part, my data collection strategy is descriped in details in section 2.4. . 

h) I also used the Generators, however I was running all my experiments on a GPU from AWS (g2.2xlarge), I was always running out of space. Generators was really helpful in that.
 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####  **Summary**
My code is using  Keras with tensorflow, using the follwoing steps:

a) Converting images from BGR to YUV
b) Combined Images from Central, left and right Cameras with steering correction angle = 0.2
c) I augmented more data, as I used Images from the Centeral Camera, flipping it then reverse the corresponding steering angle (*-1.0).
d) Cropping 60-pixels from the top and 20-pixels from the bottom to not be distracted by un-important details.
e) I used Generators, to overcome the space limitation problem
f) Normalizing the images, using lambda layer.
g Applying five convolutional layers
h) Applying four fully-connected layers
i) Training the network over 40,984 images, 20% is splited as validation dataset, my data collected as described in details in the writeup.md.

#### 3.2. Final Model Architecture
**Question9: Is the model architecture documented?**

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 
Look at section 2.1.



#### 3.3. Creation of the Training Set & Training Process
**Question10: Is the creation of the training dataset and training process documented?**

look at section 2.4. 


### 4. Simulation
**Question11:Is the car able to navigate correctly on test data?**

yes, as shown in the video.mp4, which shows the track1-autonomous drive on a full lap.


## Future work
If I had more time for this project, I would have collect more data from track1 and track2 to generalize my model, also would collect more counter clockwise tracks data. I also suggest to use transfer learning, so I can reduce the training time.
