import os
import model as modellib
import FashionAI


DATA_DIR = "/data/xiangyaojun/data_sets/FashionAI"

annotations_path = {
    "train_all": os.path.join(DATA_DIR, "train/Annotations/train_all.csv"),
    "warmup_train": os.path.join(DATA_DIR, "train/Annotations/warmup_train.csv"),
    "val": os.path.join(DATA_DIR, "train/Annotations/val.csv"),
}
type_list = [["blouse", "skirt"],["outwear","trousers"],["dress"],["blouse", "dress", "outwear", "skirt", "trousers"]]
# ["blouse","dress","outwear","skirt","trousers"]
valid_type = type_list[1]
print(valid_type)
FINE_TUNE = True

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


config = FashionAI.FashionConfig()
config.display()

# Training dataset
train_csv_path = annotations_path["train_all"]
train_dataset_keypoints = FashionAI.FashionDataset(train_csv_path, DATA_DIR+"/train/")
train_dataset_keypoints.load_fashions(valid_type)
train_dataset_keypoints.prepare()

#Validation dataset
val_csv_path = annotations_path["val"]
val_dataset_keypoints = FashionAI.FashionDataset(val_csv_path, DATA_DIR+"/train/")
val_dataset_keypoints.load_fashions(valid_type)
val_dataset_keypoints.prepare()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

if FINE_TUNE:
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs//mask_rcnn_ROI_SINGLE_0050_o_t_5.595.h5")
    # model.load_weights(COCO_MODEL_PATH,by_name=True,exclude=["mrcnn_keypoint_mask_conv1"])
    model.load_weights(COCO_MODEL_PATH, by_name=True)
else:
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_coco.h5")
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_keypoint_mask", "mrcnn_bbox_fc",
                                                               "mrcnn_bbox", "mrcnn_mask"])
print("Loading weights from ", COCO_MODEL_PATH)

if not FINE_TUNE:
    model.train(train_dataset_keypoints, val_dataset_keypoints,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                layers='all')

model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=config.LEARNING_RATE,
            epochs=5,
            layers='heads')

model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=config.LEARNING_RATE/10,
            epochs=5,
            layers='all')

model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=config.LEARNING_RATE/100,
            epochs=100,
            layers='all')
