import cv2
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import numpy as np

# Path to your ROS bag and output video file
bag_file = '../bags/2024_1025_1502/2024-10-25-15-02-56.bag'
output_video = 'test.mp4'

# ROS setup
bridge = CvBridge()
bag = rosbag.Bag(bag_file, 'r')

# Get video properties for saving
frame_width, frame_height = 640, 480  # Change as per your image resolution
fps = 10  # Modify based on your recording

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Iterate through the ROS bag and save images to video
try:
    for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw/compressed']):
        if topic == '/camera/color/image_raw/compressed':
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Write frame to video
            video_writer.write(frame)

finally:
    bag.close()
    video_writer.release()
    print("Video saved to:", output_video)
