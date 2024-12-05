import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Register Split Datasets
register_coco_instances(
    "cable_added_train_split_train", {},
    "../cable_dataset/train_added/coco_split_train.json",
    "../cable_dataset/train_added_split/train"
)

register_coco_instances(
    "cable_added_train_split_valid", {},
    "../cable_dataset/train_added/coco_split_valid.json",
    "../cable_dataset/train_added_split/valid"
)

class TrainerWithEvaluation(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Configure Training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("cable_added_train_split_train",)
cfg.DATASETS.TEST = ("cable_added_train_split_valid",)
cfg.DATALOADER.NUM_WORKERS = 8
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 2000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Adjust for your specific use case
cfg.OUTPUT_DIR = "./output_single_stage"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = TrainerWithEvaluation(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluate the Final Model
evaluator = COCOEvaluator("cable_added_train_split_valid", cfg, False, output_dir="./output_single_stage/")
val_loader = build_detection_test_loader(cfg, "cable_added_train_split_valid")
print("Final Evaluation Results:")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
