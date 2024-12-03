import os
import cv2
import rosbag
from cv_bridge import CvBridge

# Input ROS bag file
bag_file = "your_rosbag.bag"

# Output folder for images
output_folder = "extracted_images"
os.makedirs(output_folder, exist_ok=True)

# Topic to extract images from
image_topic = "/camera/color/image_raw/compressed"

# Initialize CvBridge
bridge = CvBridge()

# Sampling frequency (every nth message)
sampling_rate = 3

# Open the ROS bag
bag = rosbag.Bag(bag_file, "r")

# Variables to track message index
message_index = 0
saved_image_index = 0

try:
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        if message_index % sampling_rate == 0:
            # Convert compressed image to OpenCV format
            try:
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as e:
                print(f"Error converting image: {e}")
                continue

            # Create image file name
            image_filename = os.path.join(output_folder, f"cable_{saved_image_index + 1}.jpg")

            # Save image
            cv2.imwrite(image_filename, cv_image)
            print(f"Saved: {image_filename}")
            saved_image_index += 1

        message_index += 1
finally:
    bag.close()

print("Image extraction complete!")
