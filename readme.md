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

After build finish, all setup complete! Check how to run below for turn on D415 and record Rosbag.
# How to run 
