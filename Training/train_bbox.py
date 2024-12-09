import os
import torch
import argparse
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


def parse_args():
    """
    Parse command-line arguments for training configurations.

    Returns:
        argparse.Namespace: Parsed arguments including:
            - train_json: Path to the first training annotations (COCO format).
            - train_images: Path to the first training image folder.
            - valid_json: Path to the first validation annotations.
            - valid_images: Path to the first validation image folder.
            - split_train_json: Path to the second training annotations.
            - split_train_images: Path to the second training image folder.
            - split_valid_json: Path to the second validation annotations.
            - split_valid_images: Path to the second validation image folder.
            - batch_size: Images per batch for training.
            - learning_rate: Optimizer learning rate.
            - max_iter_stage1: Training iterations for stage 1.
            - max_iter_stage2: Training iterations for stage 2.
            - num_classes: Number of object classes in the dataset.
    """
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model for cable detection.")
    parser.add_argument("--train_json", type=str, default="../cable_dataset/train/train_annotations.coco.json")
    parser.add_argument("--train_images", type=str, default="../cable_dataset/train")
    parser.add_argument("--valid_json", type=str, default="../cable_dataset/valid/valid_annotations.coco.json")
    parser.add_argument("--valid_images", type=str, default="../cable_dataset/valid")
    parser.add_argument("--split_train_json", type=str, default="../cable_dataset/train_added/coco_split_train.json")
    parser.add_argument("--split_train_images", type=str, default="../cable_dataset/train_added_split/train")
    parser.add_argument("--split_valid_json", type=str, default="../cable_dataset/train_added/coco_split_valid.json")
    parser.add_argument("--split_valid_images", type=str, default="../cable_dataset/train_added_split/valid")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--max_iter_stage1", type=int, default=3000)
    parser.add_argument("--max_iter_stage2", type=int, default=2000)
    parser.add_argument("--num_classes", type=int, default=7)
    return parser.parse_args()


class TrainerWithEvaluation(DefaultTrainer):
    """
    Custom Trainer class that extends DefaultTrainer with COCO evaluation.

    Methods:
        build_evaluator(cls, cfg, dataset_name, output_folder=None):
            Builds and returns a COCOEvaluator for evaluation.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create a COCOEvaluator for evaluating the model.

        Args:
            cfg (detectron2.config.CfgNode): Model configuration.
            dataset_name (str): Name of the validation dataset.
            output_folder (str, optional): Directory for evaluation results.

        Returns:
            COCOEvaluator: Evaluator object for validation.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def main():
    """
    Main function to train the model on two stages, save models, and evaluate results.

    Steps:
        1. Register datasets for training and validation.
        2. Configure the model for the first stage and train it.
        3. Fine-tune the model on the second stage dataset.
        4. Save models and run evaluations.
    """
    args = parse_args()

    # Step 1: Register datasets
    register_coco_instances("cable_detection_train", {}, args.train_json, args.train_images)
    register_coco_instances("cable_detection_valid", {}, args.valid_json, args.valid_images)
    register_coco_instances("cable_added_train_split_train", {}, args.split_train_json, args.split_train_images)
    register_coco_instances("cable_added_train_split_valid", {}, args.split_valid_json, args.split_valid_images)

    # Step 2: Configure the model for the first stage
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("cable_detection_train",)
    cfg.DATASETS.TEST = ("cable_detection_valid",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iter_stage1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = "./output_stage1"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Train the model on the first dataset
    trainer = TrainerWithEvaluation(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Save the trained model
    torch.save(trainer.model.state_dict(), "./output_stage1/first_stage_model.pth")

    # Step 3: Fine-tune on the second dataset
    cfg.DATASETS.TRAIN = ("cable_added_train_split_train",)
    cfg.DATASETS.TEST = ("cable_added_train_split_valid",)
    cfg.MODEL.WEIGHTS = "./output_stage1/first_stage_model.pth"
    cfg.SOLVER.MAX_ITER = args.max_iter_stage2
    cfg.OUTPUT_DIR = "./output_stage2"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Train the model on the second dataset
    trainer = TrainerWithEvaluation(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Step 4: Evaluate the model on the second validation dataset
    evaluator = COCOEvaluator("cable_added_train_split_valid", cfg, False, output_dir="./output_stage2/")
    val_loader = build_detection_test_loader(cfg, "cable_added_train_split_valid")
    print("Final Evaluation Results:")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))


if __name__ == "__main__":
    main()
