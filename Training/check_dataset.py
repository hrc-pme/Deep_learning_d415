import os
import cv2
import json
import argparse
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode

def main(json_file, image_root):
    """
    This script is used to visualize and verify COCO-style dataset annotations. 
    The user can input the path to the COCO-style JSON annotations file and the corresponding image root directory. 
    By default, it uses preset paths for `json_file` and `image_root`.

    Usage:
        python3 check_dataset.py --json_file <path_to_annotations_file> --image_root <path_to_image_directory>

    Parameters:
    -----------
    - `json_file` (str): Path to the COCO-style JSON annotations file. Default: "../cable_dataset/train/train_annotations.coco.json".
    - `image_root` (str): Path to the image root directory. Default: "../cable_dataset/train/".

    Interactive Features:
    ---------------------
    - Press the `Space` key to move to the next image.
    - Press the `Q` key to quit the visualization.

    Example:
    --------
    python3 check_dataset.py --json_file "../cable_dataset/train_added/coco_v2.json" --image_root "../cable_dataset/train_added/2024_1202"
    """

    # Load COCO-style annotations
    with open(json_file) as f:
        coco_data = json.load(f)

    # Create a mapping from image ids to image file names
    image_dict = {image['id']: image['file_name'] for image in coco_data['images']}

    # Create a mapping from category ids to category names
    category_dict = {category['id']: category['name'] for category in coco_data['categories']}

    # Test the dataset by iterating through the images and annotations
    annotations = coco_data['annotations']
    
    # Group annotations by image_id
    image_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    print(f"Number of samples in dataset: {len(image_annotations)}")
    
    # Visualize samples
    for image_id, anns in image_annotations.items():
        img_path = os.path.join(image_root, image_dict[image_id])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image {img_path}")
            continue
        visualizer = Visualizer(img[:, :, ::-1], scale=0.5)

        print(f"\nAnnotations for image {image_dict[image_id]} (ID: {image_id}):")
        print(f"Number of annotations: {len(anns)}")
        if len(anns) > 0:
            for ann in anns:
                print(f"  Bounding Box: {ann['bbox']}, Category ID: {ann['category_id']}")
        
        # Convert COCO bounding boxes to absolute format [x1, y1, x2, y2]
        bbox_annotations_labelme = [
            {
                "bbox": [
                    ann["bbox"][0],  # upper_left_x
                    ann["bbox"][1],  # upper_left_y
                    ann["bbox"][0] + ann["bbox"][2],  # lower_right_x (x + width)
                    ann["bbox"][1] + ann["bbox"][3],  # lower_right_y (y + height)
                ],
                "bbox_mode": BoxMode.XYXY_ABS,  
                "category_id": ann["category_id"],
            }
            for ann in anns
        ]

        # Visualize with corrected bounding boxes
        out = visualizer.overlay_instances(
            boxes=[x["bbox"] for x in bbox_annotations_labelme],
            labels=[category_dict[x["category_id"]] for x in bbox_annotations_labelme]
        )
        cv2.imshow("Dataset Sample", out.get_image()[:, :, ::-1])
        
        # Wait for user input
        key = cv2.waitKey(0)
        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord(' '):  # Next image on 'space'
            continue

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COCO-style annotations.")
    parser.add_argument("--json_file", type=str, default="../cable_dataset/train/train_annotations.coco.json",
                        help="Path to the COCO-style JSON annotations file.")
    parser.add_argument("--image_root", type=str, default="../cable_dataset/train/",
                        help="Path to the root directory containing the images.")
    args = parser.parse_args()

    main(args.json_file, args.image_root)
