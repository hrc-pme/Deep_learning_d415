#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class RealTimeInferenceNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node("realtime_detection_node")

        # Setup Detectron2 config
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

        # ROS Image and bridge setup
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback)
        
        # Publisher for annotated output
        self.detection_pub = rospy.Publisher("/camera/color/image_detection/compressed", CompressedImage, queue_size=1)
        
        # Timer for inference at 1 Hz
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback)  # 1 Hz timer

        # Variables for performance evaluation
        self.current_image = None
        self.inference_times = []
        self.start_time = None

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.current_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def timer_callback(self, event):
        if self.current_image is not None:
            # Start timing
            self.start_time = time.time()
            
            # Run inference on the latest image
            outputs = self.predictor(self.current_image)
            # Draw bounding boxes, masks, and labels
            annotated_image = self.draw_detections(outputs)

            # Stop timing
            end_time = time.time()
            inference_duration = end_time - self.start_time
            self.inference_times.append(inference_duration)

            # Print current and average inference time
            avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            rounded_duration = round(inference_duration, 1)
            rospy.loginfo(f"Inference Time: {inference_duration:.4f} s, you can set your timer duration to {rounded_duration:.1f} in code")

            # Convert annotated image to CompressedImage message and publish
            self.publish_detection(annotated_image)

    def draw_detections(self, outputs):
        # Draw bounding boxes, masks, and labels
        v = Visualizer(self.current_image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return v.get_image()[:, :, ::-1]

    def publish_detection(self, annotated_image):
        # Convert the image to CompressedImage
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', annotated_image)[1]).tobytes()

        # Publish the compressed annotated image
        self.detection_pub.publish(msg)

if __name__ == "__main__":
    node = RealTimeInferenceNode()
    rospy.spin()
