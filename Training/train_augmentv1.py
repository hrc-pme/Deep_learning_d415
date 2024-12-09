import os
import cv2
import copy
import argparse
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.structures import BoxMode


def parse_args():
    """
    Parse command-line arguments for training configurations.

    Returns:
        argparse.Namespace: Parsed arguments including:
            - train_json: Path to training annotations (COCO format).
            - train_images: Path to training image folder.
            - valid_json: Path to validation annotations.
            - valid_images: Path to validation image folder.
            - batch_size: Images per batch for training.
            - learning_rate: Optimizer learning rate.
            - max_iter: Maximum training iterations.
            - num_classes: Number of object classes in the dataset.
    """
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model for cable segmentation.")
    parser.add_argument("--train_json", type=str, default="../cable_dataset/train/train_annotations.coco.json")
    parser.add_argument("--train_images", type=str, default="../cable_dataset/train")
    parser.add_argument("--valid_json", type=str, default="../cable_dataset/valid/valid_annotations.coco.json")
    parser.add_argument("--valid_images", type=str, default="../cable_dataset/valid")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--num_classes", type=int, default=7)
    return parser.parse_args()


def register_datasets(train_json, train_images, valid_json, valid_images):
    """
    Register the training and validation datasets in COCO format.

    Args:
        train_json (str): Path to training annotations (COCO format).
        train_images (str): Path to training image folder.
        valid_json (str): Path to validation annotations (COCO format).
        valid_images (str): Path to validation image folder.
    """
    register_coco_instances("cable_segmentation_train", {}, train_json, train_images)
    register_coco_instances("cable_segmentation_valid", {}, valid_json, valid_images)


class AugmentedDatasetMapper(DatasetMapper):
    """
    Custom DatasetMapper class with augmentations for training.
    """
    def __init__(self, is_train=True):
        super().__init__(is_train)

    def __call__(self, dataset_dict):
        """
        Apply augmentations to dataset images and annotations.

        Args:
            dataset_dict (dict): Dataset entry containing image and annotations.

        Returns:
            list: Augmented dataset with multiple transformed versions of the original image.
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = cv2.imread(dataset_dict["file_name"])
        annotations = dataset_dict.get("annotations", [])
        height, width = image.shape[:2]

        # Prepare augmented images and annotations
        augmented_images = [image]
        augmented_annotations = [annotations]

        # Apply horizontal flip
        flipped_image = cv2.flip(image, 1)
        flipped_annotations = self.transform_annotations_flip(annotations, width)
        augmented_images.append(flipped_image)
        augmented_annotations.append(flipped_annotations)

        # Apply 90-degree clockwise rotation
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_annotations = self.transform_annotations_rotate(annotations, height, width, clockwise=True)
        augmented_images.append(rotated_image)
        augmented_annotations.append(rotated_annotations)

        # Convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        augmented_images.append(grayscale_image)
        augmented_annotations.append(annotations)

        # Return all augmented images
        expanded_dataset = []
        for img, anns in zip(augmented_images, augmented_annotations):
            new_entry = copy.deepcopy(dataset_dict)
            new_entry["image"] = img
            new_entry["annotations"] = anns
            expanded_dataset.append(new_entry)

        return expanded_dataset

    @staticmethod
    def transform_annotations_flip(annotations, width):
        """
        Flip bounding boxes, segmentation, and keypoints horizontally.

        Args:
            annotations (list): List of annotations containing bbox, segmentation, and keypoints.
            width (int): Width of the image.

        Returns:
            list: Transformed (flipped) annotations.
        """
        flipped_annotations = []
        for ann in annotations:
            bbox = ann["bbox"]
            x_min, y_min, box_width, box_height = bbox
            x_min_new = width - (x_min + box_width)
            flipped_bbox = [x_min_new, y_min, box_width, box_height]
            ann["bbox"] = flipped_bbox

            if "segmentation" in ann:
                ann["segmentation"] = [
                    [width - x if i % 2 == 0 else y for i, (x, y) in enumerate(zip(seg[::2], seg[1::2]))]
                    for seg in ann["segmentation"]
                ]
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                flipped_keypoints = []
                for i in range(0, len(keypoints), 3):
                    x, y, visibility = keypoints[i:i + 3]
                    flipped_keypoints.extend([width - x, y, visibility])
                ann["keypoints"] = flipped_keypoints

            flipped_annotations.append(ann)
        return flipped_annotations

    @staticmethod
    def transform_annotations_rotate(annotations, height, width, clockwise=True):
        """
        Rotate bounding boxes, segmentation, and keypoints.

        Args:
            annotations (list): List of annotations.
            height (int): Height of the image.
            width (int): Width of the image.
            clockwise (bool): If True, rotate clockwise. If False, rotate counterclockwise.

        Returns:
            list: Transformed (rotated) annotations.
        """
        rotated_annotations = []
        for ann in annotations:
            bbox = ann["bbox"]
            x_min, y_min, box_width, box_height = bbox
            if clockwise:
                x_min_new, y_min_new = height - (y_min + box_height), x_min
            else:
                x_min_new, y_min_new = y_min, width - (x_min + box_width)
            rotated_bbox = [x_min_new, y_min_new, box_height, box_width]
            ann["bbox"] = rotated_bbox

            if "segmentation" in ann:
                ann["segmentation"] = [
                    [height - y if clockwise else y, x if clockwise else width - x]
                    for seg in ann["segmentation"]
                    for x, y in zip(seg[::2], seg[1::2])
                ]
            if "keypoints" in ann:
                keypoints = ann["keypoints"]
                rotated_keypoints = []
                for i in range(0, len(keypoints), 3):
                    x, y, visibility = keypoints[i:i + 3]
                    if clockwise:
                        x_new, y_new = height - y, x
                    else:
                        x_new, y_new = y, width - x
                    rotated_keypoints.extend([x_new, y_new, visibility])
                ann["keypoints"] = rotated_keypoints

            rotated_annotations.append(ann)
        return rotated_annotations


class TrainerWithEvaluation(DefaultTrainer):
    """
    Custom Trainer class with COCO evaluation.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create a COCO evaluator for the validation dataset.

        Args:
            cfg (CfgNode): Model configuration.
            dataset_name (str): Dataset name (validation).
            output_folder (str, optional): Directory for evaluation results.

        Returns:
            COCOEvaluator: Evaluation object.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def main():
    """
    Main function for training and evaluating the model.
    """
    args = parse_args()

    # Step 1: Register the datasets
    register_datasets(args.train_json, args.train_images, args.valid_json, args.valid_images)

    # Step 2: Configure the model and training settings
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("cable_segmentation_train",)
    cfg.DATASETS.TEST = ("cable_segmentation_valid",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Step 3: Train the model
    trainer = TrainerWithEvaluation(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Step 4: Evaluate the model
    evaluator = COCOEvaluator("cable_segmentation_valid", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "cable_segmentation_valid")
    print("Validation Results:")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))


if __name__ == "__main__":
    main()
