# Training

This training folder contains example code for preparing the dataset and training a model using the Detectron2 framework.

---

## Script Description

### Dataset Preparation

The dataset must be in COCO format to be compatible with the Detectron2 framework. Currently, this script supports **bounding boxes only**. Segmentation support is planned for a future update. *(Last updated: 2024/12/06)*

1. **`labelmecoco.py`: Convert Labelme annotations to COCO format**
   This script converts a folder of Labelme JSON files into a COCO-compatible JSON file. The generated COCO JSON file includes the images, bounding boxes, and categories required by the Detectron2 framework.

   **Usage**:
   ```bash
   python3 labelmecoco.py --labelme_folder /path/to/labelme_folder --output_json /path/to/output/coco.json
   ```

   **Parameters**:
   - `--labelme_folder`: Path to the folder containing Labelme JSON files. Default is `./labelme_data`.
   - `--output_json`: Path to save the generated COCO JSON file. Default is `./output/coco.json`.

   **Example**:
   ```bash
   python3 labelmecoco.py --labelme_folder ./labelme_data --output_json ./output/coco.json
   ```

   **Note**:
   - Ensure that the `categories` in the script match the labels in your Labelme dataset.
   - If a Labelme annotation contains a label not listed in the `categories`, the script will raise an error.

---

2. **`check_dataset.py`: Verify converted COCO dataset**

   This script checks the integrity of the generated COCO dataset, ensuring all labels, bounding boxes, and image metadata are correct.

   **Usage**:
   ```bash
   python3 check_dataset.py --coco_json /path/to/output/coco.json
   ```

   **Parameters**:
   - `--coco_json`: Path to the COCO JSON file you want to verify.

   **Example**:
   ```bash
   python3 check_dataset.py --coco_json ./output/coco.json
   ```

   **Functionality**:
   - Prints a summary of the dataset (number of images, annotations, and categories).
   - Detects missing or invalid bounding boxes.
   - Ensures all category labels are defined in the COCO categories list.

   **Navigation**:
   - Press SPACE to move to the next image.
   - Press 'Q' to quit the visualization.

---
3. **`check_label.py`: Verify original labelme dataset**

   This script visualizes the annotations from Labelme JSON files overlaid on their corresponding images, allowing quick inspection and debugging.

   **Usage**:
   ```bash
   python3 visualize_labelme.py --dataset_root <path_to_dataset_folder>
   ```

   **Parameters**:
   - `--dataset_root`: Path to the folder containing JSON and image files. Default is ../cable_dataset/train_added/2024_1202.

   **Example**:
   ```bash
   python3 visualize_labelme.py --dataset_root ../cable_dataset/train_added/2024_1202
   ```

   **Functionality**:
   - Overlays bounding boxes and polygons on images.
   - Displays the label for each annotation.

   **Navigation**:
   - Press SPACE to move to the next image.
   - Press 'Q' to quit the visualization.

---
4. **`split_coco.py`: Split COCO Dataset into Training and Validation Sets**

   This script splits a COCO dataset into training and validation sets based on a specified train-validation ratio.  
   It creates new COCO JSON files for each split and organizes the corresponding images into separate directories.

   **Usage**:
   ```bash
   python3 split_coco.py --input_coco_file <path_to_input_coco_json> \
                         --images_folder <path_to_images_folder> \
                         --output_train_file <path_to_save_train_json> \
                         --output_valid_file <path_to_save_valid_json> \
                         --output_train_images <path_to_train_images_folder> \
                         --output_valid_images <path_to_valid_images_folder> \
                         --train_ratio <train_validation_ratio>
   ```

   **Parameters**:
   - `--input_coco_file`: Path to the input COCO JSON file. Default is ../cable_dataset/train_added/coco_v2.json.
   - `--images_folder`: Directory containing the dataset images. Default is ../cable_dataset/train_added/2024_1202.
   - `--output_train_file`: Path to save the training COCO JSON file. Default is ../cable_dataset/train_added/coco_split_train.json.
   - `--output_valid_file`: Path to save the validation COCO JSON file. Default is ../cable_dataset/train_added/coco_split_valid.json.
   - `--output_train_images`: Directory to save training images. Default is ../cable_dataset/train_added_split/train.
   - `--output_valid_images`: Directory to save validation images. Default is ../cable_dataset/train_added_split/valid.
   - `--train_ratio`: Ratio of images to use for training (between 0 and 1). Default is 0.8.

   **Example**:
   ```bash
   python3 split_coco.py --input_coco_file ../cable_dataset/train_added/coco_v2.json \
                         --images_folder ../cable_dataset/train_added/2024_1202 \
                         --output_train_file ../cable_dataset/train_added/coco_split_train.json \
                         --output_valid_file ../cable_dataset/train_added/coco_split_valid.json \
                         --output_train_images ../cable_dataset/train_added_split/train \
                         --output_valid_images ../cable_dataset/train_added_split/valid \
                         --train_ratio 0.8
   ```

   **Functionality**:
   - Splits the COCO dataset images into training and validation sets based on the specified ratio.
   - Copies images to their respective directories for each split.
   - Filters and saves annotations for training and validation splits.
---

## Dataset Download

We use the **cable dataset** as an example for fine-tuning. You can download the dataset from [cable_dataset](https://terabox.com/s/1wrH4PxaigWWPP-yZ2x8pAw).

---

## Training Scripts

1. **`train.py`: Fine-tuning a segmentation model on the base dataset**
   - This script uses the base dataset to fine-tune the **"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x"** model.
   - **Features**:
     - Simple fine-tuning with default parameters for segmentation tasks.
     - Includes evaluation and inference on a test image.
   - **Run**:
     ```bash
     python3 train.py --train_json <path_to_train_annotations> \
                      --train_images <path_to_train_images> \
                      --valid_json <path_to_valid_annotations> \
                      --valid_images <path_to_valid_images> \
                      --batch_size 2 \
                      --learning_rate 0.00025 \
                      --max_iter 3000 \
                      --num_classes 6 \
                      --test_image <path_to_test_image>
     ```
   - **Default Parameters**:
     - Training JSON: `../cable_dataset/train/train_annotations.coco.json`
     - Validation JSON: `../cable_dataset/valid/valid_annotations.coco.json`
     - Batch Size: `2`, Learning Rate: `0.00025`, Max Iterations: `3000`

---

2. **`train_augmentv1.py`: Fine-tuning with dataset augmentation**
   - This script enhances the training dataset by applying various augmentation techniques, including:
     - Horizontal flipping
     - 90-degree rotations
     - Grayscale conversions
   - **Purpose**:
     - Augment dataset diversity to improve model robustness.
   - **Run**:
     ```bash
     python3 train_augmentv1.py --train_json <path_to_train_annotations> \
                                --train_images <path_to_train_images> \
                                --valid_json <path_to_valid_annotations> \
                                --valid_images <path_to_valid_images> \
                                --batch_size 2 \
                                --learning_rate 0.00025 \
                                --max_iter 3000 \
                                --num_classes 7
     ```
   - **Default Parameters**:
     - Training JSON: `../cable_dataset/train/train_annotations.coco.json`
     - Batch Size: `2`, Learning Rate: `0.00025`, Max Iterations: `3000`

---

3. **`train_bbox.py`: Multi-stage fine-tuning for bounding box detection**
   - This script trains a Faster R-CNN model in two stages:
     - **Stage 1**: Fine-tune on the base dataset.
     - **Stage 2**: Further fine-tune on an additional dataset to improve bounding box detection.
   - **Run**:
     ```bash
     python3 train_bbox.py --train_json <path_to_base_train_annotations> \
                           --train_images <path_to_base_train_images> \
                           --valid_json <path_to_base_valid_annotations> \
                           --valid_images <path_to_base_valid_images> \
                           --split_train_json <path_to_split_train_annotations> \
                           --split_train_images <path_to_split_train_images> \
                           --split_valid_json <path_to_split_valid_annotations> \
                           --split_valid_images <path_to_split_valid_images> \
                           --batch_size 2 \
                           --learning_rate 0.00025 \
                           --max_iter_stage1 3000 \
                           --max_iter_stage2 2000 \
                           --num_classes 7
     ```
   - **Features**:
     - Saves the model after Stage 1 for reusability.
     - Uses additional datasets to improve model performance in Stage 2.

---

4. **`train_bbox_v2.py`: Single-stage bounding box fine-tuning**
   - This script fine-tunes a Faster R-CNN model using a single labeled dataset with bounding boxes only.
   - **Run**:
     ```bash
     python3 train_bbox_v2.py --train_json <path_to_train_annotations> \
                              --train_images <path_to_train_images> \
                              --valid_json <path_to_valid_annotations> \
                              --valid_images <path_to_valid_images> \
                              --batch_size 4 \
                              --learning_rate 0.001 \
                              --max_iter 2000 \
                              --num_classes 7
     ```
   - **Default Parameters**:
     - Training JSON: `../cable_dataset/train_added/coco_split_train.json`
     - Batch Size: `4`, Learning Rate: `0.001`, Max Iterations: `2000`

---

## Notes
- The `num_classes` parameter must match the number of object classes in your dataset.
- Adjust `batch_size`, `learning_rate`, and `max_iter` based on your hardware and dataset size.
- Outputs (trained models and logs) are saved in the `./output` directories.
