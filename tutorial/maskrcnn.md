# Mask R-CNN Detection with Detectron2
This tutorial will guide you through setting up and running Mask R-CNN for object detection using Facebook's Detectron2 framework. Before starting, ensure that you have the latest version of the repository and Docker image and that you can turn on the camera and record a ROS bag.
# Setup Verification

## Update the Repository
Ensure your repository is up-to-date if there are new commits:
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ git pull # Pulls the latest commits to your local repository
~/Deep_learning_d415$ git submodule update --init --recursive # Updates the repository submodules
```
## Update Docker Images
To ensure the latest dependencies and libraries are installed, update your Docker image:
```
$ docker pull hrcnthu/camera_ros:ipc-20.04
```
## Build your workspace in docker
When the catkin_ws add new rospackage or new c++ code, you need to rebuild your workspace for using the new functions.
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ source gpu_run.sh # you need to make this terminal as the first terminal to get into docker
~/Deep_learning_d415# cd catkin_ws
~/Deep_learning_d415/catkin_ws# catkin build # rebuild the workspace
~/Deep_learning_d415/catkin_ws# source ~/Deep_learning_d415/set_env_param.sh
```

# How to Run (Running Inference with a Pretrained Model)

## Case 1: Real-time Detection
For real-time detection, make sure the camera is active. 

### Terminal: Start Real-time Mask R-CNN Inference
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ source docker_join.sh 
~/Deep_learning_d415# rosrun detection realtime_maskrcnn.py 
```
You’ll see a prompt displaying the recommended timer duration to optimize the inference quality:
[INFO] [Time_stamp]: Inference Time: XXX s, you can set your timer duration to OOO in code  
You can adjust the timer duration in the code at ~/Deep_learning_d415/catkin_ws/src/detection/src/realtime_maskrcnn.py on line 35.

### Terminal: Check Detection Results
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ source docker_join.sh 
~/Deep_learning_d415# source check_maskrcnn.sh
```
The detection results will display on the right-hand side.


## Case2: Using Colab for Online Resources
If you prefer to use online resources, check this Colab notebook: [maskrcnn_tutorial.ipynb](https://colab.research.google.com/drive/1z4wxg82yC-eabEtts8WfO-8EfFlobfLn?usp=sharing)  
Ensure the bag file path and name are correctly specified in the script [~/Deep_learning_d415/bags/turn_bag_to_mp4.py]
You can also update the output path and file name if needed.

### Terminal: Convert ROS Bags to MP4 Video
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ source docker_join.sh 
~/Deep_learning_d415# cd bags
~/Deep_learning_d415/bags# python3 turn_bag_to_mp4.py
```
#### Resolving File Permission Issues
If you encounter file permission issues (such as files showing a lock icon), use the chown command to change the file ownership from Docker’s root user to your admin user:
```
~$ cd ~/Deep_learning_d415/bags
~/Deep_learning_d415/bags $ sudo chown -R $USER your_file_or_folder # e.g. sudo chown -R $USER ./  
```
After running this, the lock icon should disappear.

### Upload the MP4 to Colab
Once you’ve successfully converted the bag file to an MP4 video, upload it to Colab for online inference.  
Note: Do not use "Run All"; execute each cell individually to monitor the process.  
After inference, the result video will appear on the left side of Colab. Download it to review the detection results.  