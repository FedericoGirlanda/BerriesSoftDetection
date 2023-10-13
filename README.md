# Soft Detection of Blueberries

This repository contains the code, data and plots of the project that I am implementing for openIoT Lab (FBK).

### Abstract
The OpenIoT research group has a ROS/ROS2-based mobile robot based on the JPL OpenRover open-source project. The rover carries a detachable sensing platform composed of an RGB camera and a 4 DoF robotic arm from the OpenManipulator series and is also ROS/ROS2-enabled. Using this platform, the final goal of the internship is to develop a ROS module that can use the camera to detect/count fruits. To ease the initial assessment of the developed system, an existent dataset comprising 1000+ still images of blueberries will be used. 

### Content
In the [src/python](src/python/) folder the user can find all the code necessary to generate the obtained results.
Results can be obtained by running

    python src/python/softDetection.py

The ML models that are contained in [models](models) have been created and tested in [Edge Impulse](https://studio.edgeimpulse.com).
The detection performances has been calculated by running

    python src/python/performances.py

a boolean variable named "bunchDetection" is used here to evaluate the performances of the desired task.

### Installation
The use of a virtual environment for the installation is suggested as a common good programming choice. For example, the use of *pipenv* requires the following commands

    pipenv shell
    pipenv install -r software/python/requirements.txt

### Results

<div align="center">
<img width="300" src="results/imageDetection.png">
<img width="300" src="results/bunchDetection.png">
</div>

The detection accuracy calculated for the berry detection is of the 87.4% while the one regarding the bunch detection reaches the 56.3%, considering an IoU > 0.5 as a threshold for the true detection. Increasing the IoU threshold to 0.7 the precisions become 85.9% and 37.5% for berries and bunches respectively.

### Aknowledgements
The data contained in the "dataset" folder have been taken from the [Deepblueberry](https://ieeexplore.ieee.org/abstract/document/8787818) dataset for research purposes.

<div align="center">
<img width="225" src="docs/fbkLogo.png">
</div>