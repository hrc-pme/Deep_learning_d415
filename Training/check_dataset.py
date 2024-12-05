import os
import cv2
import json
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode

# Load COCO-style annotations
json_file = "../cable_dataset/train/train_annotations.coco.json"
image_root = "../cable_dataset/train/"
# json_file = "../cable_dataset/train_augmented/train_annotations_augmented.coco.json"
# image_root = "../cable_dataset/train_augmented/"
# json_file = "../cable_dataset/train_added/coco_v2.json"
# image_root = "../cable_dataset/train_added/2024_1202"

with open(json_file) as f:
    coco_data = json.load(f)

# Create a mapping from image ids to image file names
image_dict = {image['id']: image['file_name'] for image in coco_data['images']}

# Create a mapping from category ids to category names
category_dict = {category['id']: category['name'] for category in coco_data['categories']}

# Test the dataset by iterating through the images and annotations
def test_dataset():
    annotations = coco_data['annotations']
    
    # Group annotations by image_id
    image_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    print(f"Number of samples in dataset: {len(image_annotations)}")
    
    # Visualize random samples
    for image_id, anns in list(image_annotations.items())[:-10]:  # Change 3 to the number of images to test
        img_path = os.path.join(image_root, image_dict[image_id])
        img = cv2.imread(img_path)
        visualizer = Visualizer(img[:, :, ::-1], scale=0.5)

        print(f"Annotations for image {image_dict[image_id]} (ID: {image_id}):")
        print(f"Number of annotations: {len(anns)}")
        if len(anns) > 0:
            for ann in anns:
                print(f"  Bounding Box: {ann['bbox']}, Category ID: {ann['category_id']}")
        
        # Extract only bounding boxes from annotations
        # Convert COCO [x, y, width, height] to LabelMe [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
        bbox_annotations_labelme = [
            {
                "bbox": [
                    ann["bbox"][0],  # upper_left_x
                    ann["bbox"][1],  # upper_left_y
                    ann["bbox"][0] + ann["bbox"][2],  # lower_right_x (x + width)
                    ann["bbox"][1] + ann["bbox"][3],  # lower_right_y (y + height)
                ],
                "bbox_mode": BoxMode.XYXY_ABS,  # Now it's in [upper_left_x, upper_left_y, lower_right_x, lower_right_y]
                "category_id": ann["category_id"],
            }
            for ann in anns
        ]

        # Visualize with corrected bounding boxes
        out = visualizer.overlay_instances(
            boxes=[x["bbox"] for x in bbox_annotations_labelme],  # Using the converted LabelMe-style boxes
            labels=[category_dict[x["category_id"]] for x in bbox_annotations_labelme]
        )
        cv2.imshow("Dataset Sample", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)  # Press any key to move to the next image
    cv2.destroyAllWindows()

test_dataset()
