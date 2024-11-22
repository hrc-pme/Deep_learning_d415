import os
import cv2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

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

# Step 2: Create a Custom Trainer Class with Evaluation
class TrainerWithEvaluation(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Step 3: Configure the Model and Training Settings
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
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # Change this based on your dataset (e.g., 1 class for cables)

cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Step 4: Train the Model
trainer = TrainerWithEvaluation(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Step 5: Evaluate the Model on Validation Set
evaluator = COCOEvaluator("cable_segmentation_valid", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "cable_segmentation_valid")
print("Validation Results:")
print(inference_on_dataset(trainer.model, val_loader, evaluator))

# Step 6: Inference on New Images
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Path to the final trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for predictions
predictor = DefaultPredictor(cfg)

# Test on a new image
test_image_path = "./PXL_20241119_080909346.MP.jpg"  # Replace with your test image path
im = cv2.imread(test_image_path)
outputs = predictor(im)
print("Inference Outputs:", outputs)

# Optional: Visualize the results
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

v = Visualizer(im[:, :, ::-1], metadata=train_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Result", result.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()

