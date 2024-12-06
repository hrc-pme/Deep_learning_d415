import json
import os
import random
from shutil import copyfile
import argparse

def main(input_coco_file, images_folder, output_train_file, output_valid_file, 
         output_train_images, output_valid_images, train_ratio):
    """
    Splits a COCO dataset into training and validation datasets based on the given train-validation ratio.

    Parameters:
        input_coco_file (str): Path to the input COCO JSON file.
        images_folder (str): Directory containing the dataset images.
        output_train_file (str): Path to save the training COCO JSON file.
        output_valid_file (str): Path to save the validation COCO JSON file.
        output_train_images (str): Directory to save training images.
        output_valid_images (str): Directory to save validation images.
        train_ratio (float): Ratio of images to use for training (between 0 and 1).

    Example Usage:
        python split_coco.py --input_coco_file ../cable_dataset/train_added/coco_v2.json \
                             --images_folder ../cable_dataset/train_added/2024_1202 \
                             --output_train_file ../cable_dataset/train_added/coco_split_train.json \
                             --output_valid_file ../cable_dataset/train_added/coco_split_valid.json \
                             --output_train_images ../cable_dataset/train_added_split/train \
                             --output_valid_images ../cable_dataset/train_added_split/valid \
                             --train_ratio 0.8
    """
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a COCO dataset into training and validation sets.")
    parser.add_argument("--input_coco_file", type=str, default="../cable_dataset/train_added/coco_v2.json", 
                        help="Path to the input COCO JSON file.")
    parser.add_argument("--images_folder", type=str, default="../cable_dataset/train_added/2024_1202", 
                        help="Directory containing the dataset images.")
    parser.add_argument("--output_train_file", type=str, default="../cable_dataset/train_added/coco_split_train.json", 
                        help="Path to save the training COCO JSON file.")
    parser.add_argument("--output_valid_file", type=str, default="../cable_dataset/train_added/coco_split_valid.json", 
                        help="Path to save the validation COCO JSON file.")
    parser.add_argument("--output_train_images", type=str, default="../cable_dataset/train_added_split/train", 
                        help="Directory to save training images.")
    parser.add_argument("--output_valid_images", type=str, default="../cable_dataset/train_added_split/valid", 
                        help="Directory to save validation images.")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                        help="Ratio of images to use for training (between 0 and 1).")

    args = parser.parse_args()
    main(args.input_coco_file, args.images_folder, args.output_train_file, 
         args.output_valid_file, args.output_train_images, args.output_valid_images, 
         args.train_ratio)
