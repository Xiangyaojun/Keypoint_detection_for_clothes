import os
import datetime
import numpy as np
import model as modellib
import FashionAI
import skimage.io
from fushion_two_model import read_data, get_result_error


DATA_DIR = "/data/xiangyaojun/data_sets/FashionAI/train"

annotations_path = {
    "train": os.path.join(DATA_DIR, "Annotations/warm_up.csv"),
    "test": os.path.join(DATA_DIR, "Annotations/round2_test_a.csv"),
    "val": os.path.join(DATA_DIR, "Annotations/val.csv"),
}

class InferenceConfig(FashionAI.FashionConfig):
    IMAGES_PER_GPU = 1
    ROI_MODE = "ROI_SINGLE"

test_model = "val"
check_model_num = "0030"
valid_type = ["blouse", "dress", "outwear", "skirt", "trousers"]
# valid_type = ["trousers"]
WANT_SAVE_CSV = True
STOP_COUNT = -1
head = ["image_id", "image_category", "neckline_left", "neckline_right", "center_front",
       "shoulder_left", "shoulder_right", "armpit_left", "armpit_right", "waistline_left", "waistline_right",
       "cuff_left_in", "cuff_left_out", "cuff_right_in", "cuff_right_out", "top_hem_left", "top_hem_right",
       "waistband_left", "waistband_right", "hemline_left", "hemline_right", "crotch", "bottom_left_in",
       "bottom_left_out", "bottom_right_in", "bottom_right_out"]

result_csv = [head]
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# set config
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs//mask_rcnn_ROI_MULTI_0005.h5")
test_annotations_list = np.loadtxt(annotations_path[test_model], skiprows=1, delimiter=',', dtype=bytes).astype(str)
inference_config = InferenceConfig()
inference_config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=inference_config)
assert COCO_MODEL_PATH != "", "Provide path to trained weights"

# Load weights trained on COCO_MODEL_PATH
model.load_weights(COCO_MODEL_PATH, by_name=True)
print("Loading weights from ", COCO_MODEL_PATH)

count = 0
all_valid = []
for m in range(len(test_annotations_list)):
    row_annotation = test_annotations_list[m]
    img_path = os.path.join(DATA_DIR, row_annotation[0])
    clothes_type = row_annotation[0].split("/")[1]
    if clothes_type not in valid_type:
        continue
    original_image = skimage.io.imread(img_path)
    original_image_copy = original_image.copy()

    # predict the keypoints of the original_image
    results = model.detect_keypoint([original_image], inference_config.KEYPOINT_MASK_SHAPE, augment=False)
    r = results[0]
    if WANT_SAVE_CSV:
        row = ["-1_-1_-1" for i in range(26)]
        row[0] = row_annotation[0]
        row[1] = row_annotation[1]
        all_keypoints = r['keypoints'][0]
        for i in FashionAI.clothes_index[clothes_type]:
            str_one = str(all_keypoints[i-2][0]) + "_" + str(all_keypoints[i-2][1]) + "_1"
            row[i] = str_one
        row = np.array(row).astype(str)
        result_csv.append(row)

    count += 1
    if count % 10 == 0:
        print(str(count) + "/" + str(len(test_annotations_list)))
    if STOP_COUNT > 0 and count == STOP_COUNT:
        break

if WANT_SAVE_CSV:
    result_csv = np.array(result_csv)
    save_csv_path = test_model+"_"+check_model_num+"_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"
    np.savetxt(os.path.join("submit",test_model, save_csv_path), result_csv, fmt='%s', delimiter=',')
    if test_model in ["val", "train"] and len(valid_type) == 5:
        gt_data = read_data("submit/val/val.csv")
        base_predict = read_data(os.path.join("submit",test_model, save_csv_path))
        average_error, _ = get_result_error(gt_data, base_predict)
        dst_path = os.path.join("submit",test_model, test_model+"_"+check_model_num+"_"+str(round(average_error,3))+".csv")
        os.rename(os.path.join("submit",test_model, save_csv_path), dst_path)
        print(dst_path)
    else:
        print(save_csv_path)