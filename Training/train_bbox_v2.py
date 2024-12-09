import os
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
            - train_json: Path to training annotations (COCO format).
            - train_images: Path to training image folder.
            - valid_json: Path to validation annotations.
            - valid_images: Path to validation image folder.
            - batch_size: Images per batch for training.
            - learning_rate: Optimizer learning rate.
            - max_iter: Maximum training iterations.
            - num_classes: Number of object classes in the dataset.
    """
    parser = argparse.ArgumentParser(description="Train a Faster R-CNN model for cable detection.")
    parser.add_argument("--train_json", type=str, default="../cable_dataset/train_added/coco_split_train.json")
    parser.add_argument("--train_images", type=str, default="../cable_dataset/train_added_split/train")
    parser.add_argument("--valid_json", type=str, default="../cable_dataset/train_added/coco_split_valid.json")
    parser.add_argument("--valid_images", type=str, default="../cable_dataset/train_added_split/valid")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_iter", type=int, default=2000)
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
    Main function to train the model, save models, and evaluate results.

    Steps:
        1. Register the datasets for training and validation.
        2. Configure the model for training.
        3. Train the model.
        4. Evaluate the final model.
    """
    args = parse_args()

    # Step 1: Register the datasets
    register_coco_instances("cable_added_train_split_train", {}, args.train_json, args.train_images)
    register_coco_instances("cable_added_train_split_valid", {}, args.valid_json, args.valid_images)

    # Step 2: Configure the model for training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("cable_added_train_split_train",)
    cfg.DATASETS.TEST = ("cable_added_train_split_valid",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = "./output_single_stage"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Step 3: Train the model
    trainer = TrainerWithEvaluation(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Step 4: Evaluate the model on the validation set
    evaluator = COCOEvaluator("cable_added_train_split_valid", cfg, False, output_dir="./output_single_stage/")
    val_loader = build_detection_test_loader(cfg, "cable_added_train_split_valid")
    print("Final Evaluation Results:")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))


if __name__ == "__main__":
    main()
