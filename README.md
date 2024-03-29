# RoboticsCVDL-GraspDetection

HW Guide: https://github.com/notsky23/RoboticsDL-GraspDetection/blob/master/hw6-2.pdf.<br><br>

## What is this practice about?<br>

This module is a cross between computer vision and deep learning applications in robotics. We will be using point clouds to help a robot analyze and make sense of objects. We will also be using a fully convolutional neural network with the MobileNet backbone for training. The architecture we will be using is called MobileUNet, which is a custom implementation of a U-Net style network with a pretrained MobileNetv3 backbone.<br>

In this module, we will be using point clouds to identify the object and pinpoint the location of where the robot should grasp the object. Then, we will be using a convolutional neural network to train a model in order for the robot learn and to make accurate predictions of the appropriate grasp pose for a given input image<br><br>

## Installation Instructions:

We recommend using Anaconda to set up your virtual environment, but you can use whatever method you wish.
Execute the following commands to set up the environment with conda: 
```

conda create -n hw6 python=3.8
conda activate hw6
python -m pip install -r requirements.txt
```

The libraries we will be using are:
```shell
1. tqdm
2. torch
3. torchvision
4. pybullet
5. numpy
6. matplotlib
```
<br>

## Results:<br>

Here are the results I got.<br>

The code is included in this repo.<br><br>

### Q1 - Grasp Simulator:<br>

Test Simulation:<br>
![Simulation1](https://user-images.githubusercontent.com/98131995/236622234-3b492928-a316-4fe9-8dca-d8862dba5e05.gif)<br><br>

### Q2 - Create Grasp Prediction Network:<br>

![image](https://user-images.githubusercontent.com/98131995/236622429-3f2757c6-f76b-4bd8-8edb-ef2f29547c52.png)<br><br>
<img src="https://user-images.githubusercontent.com/98131995/236622581-2d0bee8f-c4af-4674-bf83-ea9230806b92.png" width=50% height=50%><img src="https://user-images.githubusercontent.com/98131995/236622636-4fd863bf-e0bd-4d63-bf1c-f9709d6f5c81.png" width=50% height=50%><br><br>

### Q3 - Training and Evaluating Network:<br>

Training:<br>
![image](https://user-images.githubusercontent.com/98131995/236622819-b619c548-1894-4f1e-8868-3d33526c6950.png)<br><br>

After Training:<br>
![image](https://user-images.githubusercontent.com/98131995/236622932-0cda1e59-729c-4e41-a749-7e467962844b.png)     ![image](https://user-images.githubusercontent.com/98131995/236622947-ca0609ed-9ed1-409d-81e5-f5bbb29e75b3.png)<br><br>

Evaluation:<br>
![image](https://user-images.githubusercontent.com/98131995/236623342-17547337-97a9-4281-9ee8-772f65ad8f82.png)<br><br>
![image](https://user-images.githubusercontent.com/98131995/236623386-b3872ee8-d26f-4611-a5bc-2720f48425da.png)<br><br>

### Q4 - Adding Data Augmentation:<br>

![image](https://user-images.githubusercontent.com/98131995/236623484-acc76e28-f470-434d-bf3f-872187d99b56.png)<br><br>

Simulation with training and data augmentation:<br>
![Simulation 2](https://user-images.githubusercontent.com/98131995/236623606-0d81ae98-c5b0-4ac4-831b-b51508afd35a.gif)<br><br>
