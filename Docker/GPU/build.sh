#!/bin/bash

# Define variables
IMAGE_NAME="hrcnthu/camera_ros:ipc-20.04"
DOCKERFILE_PATH="Dockerfile"  # This assumes Dockerfile is in the same directory as build.sh

# Build the Docker image
docker build -f $DOCKERFILE_PATH -t $IMAGE_NAME .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image '$IMAGE_NAME' built successfully."
else
    echo "Failed to build Docker image '$IMAGE_NAME'."
fi

