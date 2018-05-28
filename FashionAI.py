import numpy as np
from config import Config
import utils, math


clothes_index = {
    "blouse": [2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16],  # 13 keypoints
    'dress': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20],  # 15 keypoints
    'outwear': [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 14 keypoints
    'skirt': [17, 18, 19, 20],  # 4 keypoints
    'trousers': [17, 18, 21, 22, 23, 24, 25]  # 7 keypoints
}

class FashionConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    def __init__(self):

        self.NUM_KEYPOINTS = 24
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    KEYPOINT_MASK_SHAPE = [128, 128]
    # ["ROI_MULTI","ROI_SINGLE","ROI_ALIGN"]
    ROI_MODE = "ROI_SINGLE"
    MAX_GT_INSTANCES = 1
    KEYPOINT_MASK_POOL_SIZE = 64
    LEARNING_RATE = 0.001
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 200
    WEIGHT_LOSS = True
    KEYPOINT_THRESHOLD = 0.000

class FashionDataset(utils.Dataset):
    def __init__(self, csv_path, data_dir_path):
        # '/Users/xiangyaojun/PycharmProjects/FashionAI/train/Annotations/annotations.csv'
        self.annotations_list = np.loadtxt(csv_path, skiprows=1, delimiter=',', dtype=bytes).astype(str)
        self.DATA_DIR = data_dir_path
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
    def load_fashions(self,type):
        # Add classes
        self.add_class("fashion", 1, "clothes")
        for m in range(len(self.annotations_list)):
            row_annotation = self.annotations_list[m]
            if row_annotation[1] not in type:
                continue
            file_path = self.DATA_DIR + row_annotation[0]
            coord_list = []
            for i in range(2, 26):
                coord_tuple = [int(j) for j in row_annotation[i].split("_")]
                coord_tuple[2] += 1  # label+1
                coord_list.append(coord_tuple)
            self.add_image(source="fashion", image_id=m, path=file_path, coord_list=coord_list)

    def load_keypoints(self, image_id):
        """Load person keypoints for the given image.

        Returns:
        key_points: num_keypoints coordinates and visibility (x,y,v)  [num_person,num_keypoints,3] of num_person
        """
        keypoints = []
        class_ids = []
        keypoint = self.image_info[image_id]["coord_list"]
        keypoint = np.reshape(keypoint, (-1, 3))
        keypoints.append(keypoint)
        class_ids.append(1)
        # Pack instance masks into an array
        keypoints = np.array(keypoints, dtype=np.int32)
        class_ids = np.array(class_ids, dtype=np.int32)
        return keypoints, class_ids


