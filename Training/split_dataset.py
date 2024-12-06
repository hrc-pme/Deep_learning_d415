import json
import os
import random
from shutil import copyfile

# Paths
input_coco_file = "../cable_dataset/train_added/coco_v2.json"
images_folder = "../cable_dataset/train_added/2024_1202"
output_train_file = "../cable_dataset/train_added/coco_split_train.json"
output_valid_file = "../cable_dataset/train_added/coco_split_valid.json"
output_train_images = "../cable_dataset/train_added_split/train"
output_valid_images = "../cable_dataset/train_added_split/valid"

# Split ratios
train_ratio = 0.8  # 80% for training, 20% for validation

# Create output directories
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(output_valid_images, exist_ok=True)

# Load the input COCO file
with open(input_coco_file, 'r') as f:
    coco_data = json.load(f)

# Shuffle and split images
images = coco_data['images']
random.shuffle(images)
split_index = int(len(images) * train_ratio)
train_images = images[:split_index]
valid_images = images[split_index:]

# Helper function to copy images
def copy_images(images_list, output_folder):
    for image in images_list:
        src = os.path.join(images_folder, image['file_name'])
        dest = os.path.join(output_folder, image['file_name'])
        copyfile(src, dest)

# Copy images to the new directories
copy_images(train_images, output_train_images)
copy_images(valid_images, output_valid_images)

# Filter annotations for each split
def filter_annotations(images_subset, annotations):
    image_ids = {image['id'] for image in images_subset}
    return [annotation for annotation in annotations if annotation['image_id'] in image_ids]

train_annotations = filter_annotations(train_images, coco_data['annotations'])
valid_annotations = filter_annotations(valid_images, coco_data['annotations'])

# Create new COCO datasets
def create_coco_dataset(images, annotations, categories):
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

train_coco_data = create_coco_dataset(train_images, train_annotations, coco_data['categories'])
valid_coco_data = create_coco_dataset(valid_images, valid_annotations, coco_data['categories'])

# Save the split COCO files
with open(output_train_file, 'w') as f:
    json.dump(train_coco_data, f, indent=4)

with open(output_valid_file, 'w') as f:
    json.dump(valid_coco_data, f, indent=4)

print(f"Dataset split completed.")
print(f"Training dataset saved to: {output_train_file}")
print(f"Validation dataset saved to: {output_valid_file}")
