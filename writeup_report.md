# **Behavioral Cloning** 

## Writeup Report

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/driving_log_sample.png "driving log"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 64. The model includes RELU layers to introduce nonlinearity (code line 66,68,70), and the data is normalized in the model using a Keras lambda layer (code line 65).

TO reduce model complexity and reduce learning time, I added two max poolings.

#### 2. Attempts to reduce overfitting in the model

In order to reduce overfitting, the model contains two dropout layers both with a value of 0.5(model.py lines 74, 76).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 19).  

The model was saved and tested in the autonemous driving mode in the simulator, and it works very well.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 

I used the pictures which were captured by the central camera, and revocer by using picutures by camera of both sides.

In order to make a good training data, I carefully drive the car in the train model, I drove a couple of cycles to make sure the data is enough.

When using the photos of the left and right cameras to train the steering angle, I will give a certain correction factor to it.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use 3 layers 3x3 convolutional layer and 4 fuuly connected layer. Because this is normally considered a good start point, but its performance is turn to be poor.

Then I tried to add more convolutional layers, but the result is still not good enough.

Then I tried the nvida model. it had a low mean squared error on the training set, but the problem is it has a high mean squared error on the validation set. This may imply a overfitting situation.

Then dropout layers are added affter fully connected layer, after this is applied, the performance is good enough.


#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


* Image cropping
* Image normalization
* Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
* Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
* Fully connected: neurons: 100, activation: ELU
* Drop out (0.5)
* Fully connected: neurons: 50, activation: ELU
* Drop out (0.5)
* Fully connected: neurons: 10, activation: ELU
* Fully connected: neurons: 1 (output)


#### 3. Creation of the Training Set & Training Process

I carefully drove the car in the trainning mode in the simulator, to make sure it is good enough, and also in order to make enough amount of data, i drove a couple of cycles and then stop. I got my driving log file and IMG folder with 15357 images.

![alt text][image2]

Another method that I used is flip the image. This method doubles the training set, and also this gives a balance of right steer and left steer data because we mostly steer to the left in the train mode.

To make full use of data, I not only used the center images but also the left and right imags. In order to be able to use the left and right images, correction factors need to add.

I randomly shuffled the data set and put 10% of the data into a validation set.

Before train

Image cropping, cut off the top and buttom unnecessary pixel, avoid confuse the model.
Image normalization

As for train

I used mean squared error for the loss function to measure how close the model predicts to the given steering angle for each image.

I used Adam optimizer for optimization.

In the autonomou mode of the simulator, the car drives on the road between 2 lane lines all the time, I think it has a good enough performance.



