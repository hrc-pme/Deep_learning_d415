import json
import cv2
import os
import argparse
import numpy as np

def visualize_annotations(json_file, image_file):
    """
    Visualize the annotations from a LabelMe JSON file on the corresponding image.

    Args:
        json_file (str): Path to the LabelMe JSON file containing annotations.
        image_file (str): Path to the image corresponding to the JSON file.

    Returns:
        bool: True to continue to the next image, False to stop the visualization.
    """
    # Load the LabelMe JSON annotations
    with open(json_file) as f:
        labelme_data = json.load(f)

    # Load the image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Unable to load image at {image_file}")
        return False

    # Loop through the shapes (annotations) in the LabelMe data
    for shape in labelme_data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        
        # Check if the annotation is a polygon or a bounding box
        if len(points) == 2:  # Bounding box (2 points)
            top_left = tuple(points[0])
            bottom_right = tuple(points[1])
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green bounding box
        else:  # Polygon
            points = points.reshape((-1, 1, 2))
            cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue polygon

        # Optionally, put the label text near the annotation
        cv2.putText(image, label, tuple(points[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the annotated image
    cv2.imshow("LabelMe Annotations", image)

    # Interactive navigation
    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):  # Quit on 'q'
            return False
        elif key == ord(' '):  # Next image on 'space'
            return True

def process_folder(folder):
    """
    Traverse a folder and visualize annotations for all images and their JSON files.

    Args:
        folder (str): Path to the folder containing image and JSON files.
    """
    # Get all files in the folder
    files = os.listdir(folder)

    # Match JSON files with corresponding images
    json_files = sorted([f for f in files if f.endswith('.json')])
    image_files = {os.path.splitext(f)[0]: f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

    # Traverse JSON files and find corresponding images
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        if base_name in image_files:
            image_path = os.path.join(folder, image_files[base_name])
            json_path = os.path.join(folder, json_file)
            print(f"Processing: {image_path} with annotations {json_path}")
            if not visualize_annotations(json_path, image_path):
                print("Stopped by user.")
                break
        else:
            print(f"No matching image found for JSON file: {json_file}")

if __name__ == "__main__":
    """
    Visualize annotations from LabelMe JSON files for all images in a specified dataset folder.

    Usage:
        python script.py --dataset_root <path_to_dataset_folder>

    Parameters:
        --dataset_root (str): Path to the root folder containing JSON and image files.
                              Default is "../cable_dataset/train_added/2024_1202".

    Navigation:
        - Press SPACE to move to the next image.
        - Press 'Q' to quit the visualization.
    """
    parser = argparse.ArgumentParser(description="Visualize LabelMe JSON annotations for all images in a folder.")
    parser.add_argument("--dataset_root", type=str, default="../cable_dataset/train_added/2024_1202",
                        help="Path to the root folder containing JSON and image files. Default is '../cable_dataset/train_added/2024_1202'.")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"Error: The folder {args.dataset_root} does not exist or is not a directory.")
    else:
        process_folder(args.dataset_root)
