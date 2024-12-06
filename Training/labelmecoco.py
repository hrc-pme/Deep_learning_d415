import os
import json
import numpy as np
import glob
from labelme import utils
from PIL import Image, ImageDraw
# How to use: python3 labelmecoco.py /path/to/labelme /path/to/output/coco.json

class Labelme2COCO:
    def __init__(self, labelme_folder, save_json_path, categories):
        """
        Convert Labelme annotations to COCO format.

        :param labelme_folder: Folder containing Labelme JSON files.
        :param save_json_path: Path to save the output COCO JSON file.
        :param categories: Predefined categories in COCO format.
        """
        self.labelme_folder = labelme_folder
        self.save_json_path = save_json_path
        self.categories = categories
        self.images = []
        self.annotations = []
        self.ann_id = 1  # Start annotation IDs from 1
        self.category_map = {cat["name"]: cat["id"] for cat in self.categories}
        self.process()

    def process(self):
        labelme_files = glob.glob(os.path.join(self.labelme_folder, "*.json"))
        for img_id, json_file in enumerate(labelme_files):
            with open(json_file, "r") as f:
                data = json.load(f)
                # Process image
                self.images.append(self.create_image_entry(data, img_id))
                # Process annotations
                for shape in data["shapes"]:
                    self.annotations.append(
                        self.create_annotation_entry(shape, img_id)
                    )

        # Save the resulting COCO dataset
        coco_format = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }
        os.makedirs(os.path.dirname(self.save_json_path), exist_ok=True)
        with open(self.save_json_path, "w") as f:
            json.dump(coco_format, f, indent=4)
        print(f"COCO JSON saved to {self.save_json_path}")

    def create_image_entry(self, data, img_id):
        img = utils.img_b64_to_arr(data["imageData"])
        height, width = img.shape[:2]
        return {
            "id": img_id,
            "file_name": os.path.basename(data["imagePath"]),
            "height": height,
            "width": width,
        }

    def create_annotation_entry(self, shape, img_id):
        points = shape["points"]
        label = shape["label"]
        category_id = self.category_map.get(label)
        if category_id is None:
            raise ValueError(f"Label '{label}' not found in predefined categories.")

        # Flatten points for segmentation
        # segmentation = [list(np.asarray(points).flatten())]
        segmentation = []

        # Calculate bounding box (min x, min y, width, height)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        width = x_max - x_min
        height = y_max - y_min
        bbox = [x_min, y_min, width, height]  # Ensure width and height are correct
        
        # Use bounding box area for the 'area' field (width * height)
        area = width * height

        self.ann_id += 1
        return {
            "id": self.ann_id,
            "image_id": img_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        }


        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Labelme JSON files to COCO JSON.")
    parser.add_argument("labelme_folder", type=str, help="Path to folder containing Labelme JSON files.")
    parser.add_argument("output_json", type=str, help="Path to output COCO JSON file.")
    args = parser.parse_args()

    # Define the categories as per your use case
    categories = [
        {"id": 0, "name": "RJ45", "supercategory": "none"},
        {"id": 1, "name": "RJ45-0", "supercategory": "RJ45"},
        {"id": 2, "name": "RJ45-1", "supercategory": "RJ45"},
        {"id": 3, "name": "power cord 0", "supercategory": "RJ45"},
        {"id": 4, "name": "power cord 1", "supercategory": "RJ45"},
        {"id": 5, "name": "pressure-connector-0", "supercategory": "RJ45"},
        {"id": 6, "name": "pressure-connector-1", "supercategory": "RJ45"},
    ]

    # Convert Labelme annotations to COCO
    Labelme2COCO(args.labelme_folder, args.output_json, categories)
