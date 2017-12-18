# Behaviorial Cloning Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


**Goals:** The aim of this project is to create an autonomous driving car behaviorial simulation, using Deep machine learning techniques including Convolutional Neural Networks (CNN). The model here is trained and validated using a collected Dataset. After the model is trained, validated and tested, I ran the simulator on my trained model and my car was successfully able to navigate autonomously in the center of the required lane, more details about implementation and design is found in the writeup.md.

**NOTE** I ran my model on a g2.2xlarge GPU, from Amazon web services (AWS), as it was impossible to train my model on a CPU, also, I used data Generators as I ran out on memory.

This is my submission for this project, which is the 3rd project in Self-driving car Nanodegree Program/Term1.


---
## **The steps of this project are the following:**
* I used the given [simulator](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/1c9f7e68-3d2c-4313-9c8d-5a9ed42583dc) to collect data of good driving behavior and added it to the Sample training data.
* I designed, trained and validated the model which predicts a steering angle from image data
* I used the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* My detailed work could be found in the writeup report


## **My project submission icludes the following files/folders:** 
* model.py : script used to create and train the model.
* drive.py : script to drive the car, (I didn't need to modify this file).
* model.h5: my trained Keras model.
* writeup.md: a report writeup file as markdown.
* video.mp4: a video recording my vehicle driving autonomously around the track for one full lap.
* examples: This folder includes all images used in the writeup report. 
* clone_test.ipynb: This is the notebook version of the code in model.py, which I used for testing over the GPU






### Dependencies
This project requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.




## How to drive the car in the simulator using my model?

### `drive.py`

run `drive.py` using my saved the trained model model.h5 file, i.e.

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.





