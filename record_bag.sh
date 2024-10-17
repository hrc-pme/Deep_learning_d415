#! /bin/bash

PREFIX=$(date +%Y-%m-%d-%H-%M-%S)

BAGS=$HOME/Deep_learning_d415/bags/$(date +%Y_%m%d_%H%M)

if [ ! -d "$BAGS" ]; then
    mkdir -m 775 $BAGS
fi

BAGS=$BAGS"/"$PREFIX
echo "BAGS: "$BAGS

rosbag record -O $BAGS \
    /camera/color/camera_info \
    /camera/color/image_raw/compressed \
    /camera/aligned_depth_to_color/image_raw/compressedDepth \
        





