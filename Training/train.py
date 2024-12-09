import os
import cv2
import argparse
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode


def parse_args():
    """
    Parse command-line arguments for configuring the training and inference process.

    Returns:
        argparse.Namespace: Parsed arguments including:
            - train_json: Path to the training annotations in COCO format.
            - train_images: Path to the folder containing training images.
            - valid_json: Path to the validation annotations in COCO format.
            - valid_images: Path to the folder containing validation images.
            - batch_size: Number of images per batch during training.
            - learning_rate: Learning rate for the optimizer.
            - max_iter: Total number of iterations for training.
            - num_classes: Number of object classes in the dataset.
            - test_image: Path to the image used for testing inference.
    """
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model for cable segmentation.")
    parser.add_argument("--train_json", type=str, default="../cable_dataset/train/train_annotations.coco.json",
                        help="Path to the COCO-format JSON file for training annotations.")
    parser.add_argument("--train_images", type=str, default="../cable_dataset/train",
                        help="Path to the training image folder.")
    parser.add_argument("--valid_json", type=str, default="../cable_dataset/valid/valid_annotations.coco.json",
                        help="Path to the COCO-format JSON file for validation annotations.")
    parser.add_argument("--valid_images", type=str, default="../cable_dataset/valid",
                        help="Path to the validation image folder.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per iteration. Higher values use more GPU memory.")
    parser.add_argument("--learning_rate", type=float, default=0.00025,
                        help="Learning rate for training. Larger values may speed up training but risk overfitting.")
    parser.add_argument("--max_iter", type=int, default=3000,
                        help="Maximum training iterations. Higher values improve model performance but increase time.")
    parser.add_argument("--num_classes", type=int, default=6,
                        help="Number of classes in the dataset.")
    parser.add_argument("--test_image", type=str, default="./PXL_20241119_080909346.MP.jpg",
                        help="Path to a test image for inference.")
    return parser.parse_args()


class TrainerWithEvaluation(DefaultTrainer):
    """
    Custom Trainer class that extends DefaultTrainer to include COCO evaluation.

    Methods:
        build_evaluator(cls, cfg, dataset_name, output_folder=None):
            Builds and returns a COCOEvaluator for evaluating the model during or after training.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Custom trainer class with evaluation support during training.

        Methods:
            build_evaluator: Create an evaluator for COCO-style evaluation.

        Attributes:
            cfg (detectron2.config.CfgNode): Detectron2 configuration.
            dataset_name (str): Name of the validation dataset.
            output_folder (str): Folder to save evaluation results.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def main():
    """
    Main function to train the model, evaluate it, and perform inference on a test image.

    Steps:
        1. Parse command-line arguments.
        2. Register training and validation datasets.
        3. Configure training parameters using Detectron2's get_cfg().
        4. Train the model using a custom trainer.
        5. Evaluate the model on the validation dataset.
        6. Perform inference on a new image and display the results.
    """
    args = parse_args()

    # Step 1: Register datasets
    register_coco_instances("cable_segmentation_train", {}, args.train_json, args.train_images)
    register_coco_instances("cable_segmentation_valid", {}, args.valid_json, args.valid_images)

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
    cfg.SOLVER.STEPS = []  # No learning rate decay for simplicity
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Step 3: Train the model
    trainer = TrainerWithEvaluation(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Step 4: Evaluate the model on the validation dataset
    evaluator = COCOEvaluator("cable_segmentation_valid", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "cable_segmentation_valid")
    print("Validation Results:")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))

    # Step 5: Inference on a new image
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set prediction threshold
    predictor = DefaultPredictor(cfg)

    # Step 6: Read the test image, run inference, and visualize the results
    im = cv2.imread(args.test_image)
    outputs = predictor(im)
    print("Inference Outputs:", outputs)

    v = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get("cable_segmentation_train"),
                   scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Result", result.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

