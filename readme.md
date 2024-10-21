# Deep Learning D415 Repo
This repo is an example for running intel realsense D415 camera in 20.04 Docker with ros.  
With this repo you can 
1. Turn on intel realsense D415 camera for RGBD image
2. Using RVIZ for checking the camera view of RGB D image 
3. Record a rosbag for further detection with MaskRCNN  

# Setup (Only need to do for first time setup)
```
$: means command use in terminal  
#: means command use in docker terminal  

```
## 1.Clone the repo
```
$ cd ~
$ git clone --recursive git@github.com:wellyowo/Deep_learning_d415.git
```
* --recursive is the must have procedure for clone the submodule


## 2.Update repo and submodules

```
$ git pull
$ git submodule sync --recursive
$ git submodule update --init --recursive
```

## 3. Build ROS workspace
* Check the device you are using (if you have gpu please use gpu_run.sh, if you are none-gpu device, use cpu_run.sh)
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ source gpu_run.sh / source cpu_run.sh
~/Deep_learning_d415# cd catkin_ws/src
 ~/Deep_learning_d415/catkin_ws/src# catkin build 
```
## 4. Check camera serial number and fill in the camera launch file (do in the same terminal after 3.)
* Please connect the D415 camera first
```
 ~/Deep_learning_d415/catkin_ws/src# cd ~/Deep_learning_d415
 ~/Deep_learning_d415/# rs-enumerate-devices -s # you can see the serial number of your D415 camera
```
open VScode and find the ~/Deep_learning_d415/catkin_ws/src/realsense-ros/realsense2_camera/launch/camera.launch and fill in the serial_no blank.

After fill in the serial number, all setup complete! Check how to run below for turn on D415 and record Rosbag.

# Update the repo
You need to update the repo if the repo have new commits.
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ git pull # This will pull the new commit to your local file.
~/Deep_learning_d415$ git submodule update --init --recursive # This will update the submodule of the repo.
```

# How to run   
## Terminal 1: Roscore
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ source gpu_run.sh / source cpu_run.sh
~/Deep_learning_d415# source set_env_param.sh # This script will setup the ros env parameter and write into .bashrc
~/Deep_learning_d415# roscore # Open a roscore
```
* you can use ctrl+shift+E / ctrl+shift+O to split the terminal horizontal / vertically

## Terminal 2: Turn on D415 
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ source docker_join.sh 
~/Deep_learning_d415# roslaunch realsense2_camera camera.launch
```


## Terminal 3: Rviz for checking the image
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ source docker_join.sh 
~/Deep_learning_d415# source check_D415.sh
```

## Terminal 4: Record a rosbag
```
$ cd ~/Deep_learning_d415
~/Deep_learning_d415$ source docker_join.sh 
~/Deep_learning_d415# source record_bag.sh
use Ctrl + C to stop recording
```

You can find the bag file in ~/Deep_learning_d415/bags.