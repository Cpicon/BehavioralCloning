# **Behavioral Cloning** 

## Writeup Template

One of the challenges in training and testing ML algorithms is to obtain high quality data: the live data your algorithms will be working on often isn’t availabe from the word go and even if it is, it may not contain all the features you want to train your algorithm on.

One approach to dealing with this is to generate data using simulations – this gives you full control both of the features contained within the data and also the volume and frequency of the data.

This project focused on obtaining the data set through a simulator. The simulator consists of a vehicle that drives on two tracks. One easier than the other. The simulator has two modes of use. The first is to obtain data to train the model. The second is to test the model trained previously. The was provided by Udacity, you can find the repository [here](https://github.com/udacity/self-driving-car-sim).

The model was based on the architecture proposed by NVIDIA in this [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The model is named End to End Learning.  

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* download_data.py containing the script to download the data
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### Using this project
First , run `download_data.py` to get data. use:
```sh
python download_data.py
```
Second, explore the data using `DataExploration.ipynb`. Use:
```sh
jupyter notebook DataExploration.ipynb
```

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The next [notebook](./DataExploration.ipynb) shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

[png](./README_IMAGES/model.png)

The model consists of 3 a convolution neural network with 5x5 filter sizes and depths between 24 and 48. And 2 a convolution neural network with 3x3 filter sizes and depths of 64. Finally, the model has 3 last fully-connected layers. 

The model includes ELU layers to introduce nonlinearity (see [paper](https://arxiv.org/pdf/1511.07289.pdf), and the data is normalized in the model using a Keras lambda layer. 

#### 2 Data augmented and Overfitting 
#### 2.1 Augmented Data

I obtained 3 images by each original image. One of them corresponds flipped image and, in the other,  the brightness was changed arbitrarily.

to flip the image, the `cv2.flip()` was used. The corresponding angle was multiplied by -1. The brightness was randomly modified to simulate lights environment conditions. To do that, the image was converted to HSV and the channel V was changed randomly. The image was turned back to RGB space.

To the left and right camera images, the same process was applied but in this case, in the corresponding angle was corrected by a factor of 0,2.

I decided to shuffle the images so that the order in which images comes doesn't matters to the CNN

#### 2..2 Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I choose Mean Squared Error (MSE) as loss function.

#### 4. Appropriate training data

I just examine the data given by Udacity([dataset](https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip)) and found that the data correspond to 9 laps in the first easy track. The data was pre-proccesed to get augmented data in order to generalize the model. First, the data was cropped from top because I did not wanted to distract the model with trees and sky and 25 pixels from the bottom so as to remove the dashboard that is coming in the images.

[png](./README_IMAGES/FinalModel.png)

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### Output Video

To test the model I recurred to autonomous mode on the simulator to get the views of the car. With the images collected I used `video.py` to make a video with the images obtained. Thus, I evaluate the performance of the model. The test video can be seen [here](https://youtu.be/R3WYhg16rT0)
