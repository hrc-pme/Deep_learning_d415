import os
import cv2
import copy
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper
from detectron2.structures import BoxMode

# Step 1: Register Datasets
# Register train dataset
register_coco_instances(
    "cable_segmentation_train", {},
    "../cable_dataset/train/train_annotations.coco.json",
    "../cable_dataset/train"
)

# Register validation dataset
register_coco_instances(
    "cable_segmentation_valid", {},
    "../cable_dataset/valid/valid_annotations.coco.json",
    "../cable_dataset/valid"
)

# Step 2: Custom Dataset Mapper with Augmentations
class AugmentedDatasetMapper(DatasetMapper):
    def __init__(self, is_train=True):
        super().__init__(is_train)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # Avoid modifying the original
        image = cv2.imread(dataset_dict["file_name"])
        annotations = dataset_dict.get("annotations", [])
        height, width = image.shape[:2]

        # Prepare augmented images and annotations
        augmented_images = []
        augmented_annotations = []

        # Original image
        augmented_images.append(image)
        augmented_annotations.append(annotations)

        # Horizontal flip
        flipped_image = cv2.flip(image, 1)
        flipped_annotations = self.transform_annotations_flip(annotations, width)
        augmented_images.append(flipped_image)
        augmented_annotations.append(flipped_annotations)

        # Rotate 90 degrees clockwise
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_annotations = self.transform_annotations_rotate(annotations, height, width, clockwise=True)
        augmented_images.append(rotated_image)
        augmented_annotations.append(rotated_annotations)

        # Convert to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)  # Ensure 3-channel
        augmented_images.append(grayscale_image)
        augmented_annotations.append(annotations)  # No changes needed for grayscale

        # Return all augmented versions
        expanded_dataset = []
        for img, anns in zip(augmented_images, augmented_annotations):
            new_entry = copy.deepcopy(dataset_dict)
            new_entry["image"] = img
            new_entry["annotations"] = anns
            expanded_dataset.append(new_entry)

        return expanded_dataset

    @staticmethod
    def transform_annotations_flip(annotations, width):
        flipped_annotations = []
        for ann in annotations:
            # Flip bounding boxes
            bbox = ann["bbox"]
            x_min, y_min, box_width, box_height = bbox
            x_min_new = width - (x_min + box_width)  # Flip x-coordinate
            flipped_bbox = [x_min_new, y_min, box_width, box_height]
            ann["bbox"] = flipped_bbox

            # Flip masks (if present)
            if "segmentation" in ann:
                ann["segmentation"] = [
                    [width - x if i % 2 == 0 else y for i, (x, y) in enumerate(zip(seg[::2], seg[1::2]))]
                    for seg in ann["segmentation"]
                ]

            # Flip keypoints (if present)
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
        rotated_annotations = []
        for ann in annotations:
            # Rotate bounding boxes
            bbox = ann["bbox"]
            x_min, y_min, box_width, box_height = bbox
            if clockwise:
                x_min_new, y_min_new = height - (y_min + box_height), x_min
            else:
                x_min_new, y_min_new = y_min, width - (x_min + box_width)
            rotated_bbox = [x_min_new, y_min_new, box_height, box_width]  # Swap width and height
            ann["bbox"] = rotated_bbox

            # Rotate masks (if present)
            if "segmentation" in ann:
                ann["segmentation"] = [
                    [height - y if clockwise else y, x if clockwise else width - x]
                    for seg in ann["segmentation"]
                    for x, y in zip(seg[::2], seg[1::2])
                ]

            # Rotate keypoints (if present)
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


# Step 3: Create a Custom Trainer Class with Evaluation
class TrainerWithEvaluation(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


# Step 4: Configure the Model and Training Settings
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("cable_segmentation_train",)
cfg.DATASETS.TEST = ("cable_segmentation_valid",)
cfg.DATALOADER.NUM_WORKERS = 4

# Adjust these parameters based on dataset size and GPU capacity
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = []  # No LR decay for simplicity
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Change this based on your dataset (e.g., 1 class for cables)

cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Step 5: Train the Model
trainer = TrainerWithEvaluation(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Step 6: Evaluate the Model on Validation Set
evaluator = COCOEvaluator("cable_segmentation_valid", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "cable_segmentation_valid")
print("Validation Results:")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
