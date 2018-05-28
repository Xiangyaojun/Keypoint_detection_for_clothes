"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import logging
from collections import OrderedDict
import numpy as np
import cv2
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))

############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i))
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementatin of Log2. TF doesn't have a native implemenation."""
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, channels]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, image_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = tuple(image_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(
            self.image_shape[0] * self.image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indicies for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1], )


############################################################
#  Detection Target Layer
############################################################

def detection_keypoint_targets_graph(proposals, gt_keypoints, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_keypoints: [MAX_GT_INSTANCES, NUM_KEYPOINTS, 3] of (x, y ,v)
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
    Returns: Target ROIs and corresponding class IDs, bounding box shifts, keypoint label, keypoint weight
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Class-specific bbox refinements.
    keypoints_labels: [TRAIN_ROIS_PER_IMAGE, NUM_KEYPOINTS). Keypoint labels in [0, HEATMAP_SIZE-1]
    HEATMAP_SIZE = HEAT_MAP_WITHD * HEAT_MAP_HEIGHT

    keypoints_weights: [TRAIN_ROIS_PER_IMAGE, NUM_KEYPOINTS), 0: not visible 1: visible

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Pick the right mask for each ROI
    roi_keypoints = gt_keypoints

    # Transform ROI keypoints from (x,y) image space to keypoint label
    y1, x1, y2, x2 = tf.split(proposals, 4, axis=1)
    y1 = y1[:, 0]
    x1 = x1[:, 0]
    y2 = y2[:, 0]
    x2 = x2[:, 0]
    scale_x = tf.cast(config.KEYPOINT_MASK_SHAPE[1] / ((x2 - x1) * config.IMAGE_SHAPE[1]), tf.float32)
    scale_y = tf.cast(config.KEYPOINT_MASK_SHAPE[0] / ((y2 - y1) * config.IMAGE_SHAPE[0]), tf.float32)
    keypoint_labels = []
    keypoint_weights = []


    for k in range(config.NUM_KEYPOINTS):
        vis = roi_keypoints[:, k, 2] > 0
        x = tf.cast(roi_keypoints[:, k, 0], tf.float32)
        y = tf.cast(roi_keypoints[:, k, 1], tf.float32)

        #  recover from normlized corrdinates to real word
        x_real = (x - x1) * config.IMAGE_SHAPE[1]
        y_real = (y - y1) * config.IMAGE_SHAPE[0]
        #  transform the box size into feature map size = KEYPOINT_MASK_SHAPE
        x_real_map = tf.cast(x_real * scale_x + 0.5, tf.int32)
        y_real_map = tf.cast(y_real * scale_y + 0.5, tf.int32)
        x_boundary_bool = tf.cast((x_real_map == config.KEYPOINT_MASK_SHAPE[1]), tf.int32)
        y_boundary_bool = tf.cast((y_real_map == config.KEYPOINT_MASK_SHAPE[1]), tf.int32)
        y_real_map = y_real_map * (1 - y_boundary_bool) + y_boundary_bool * (config.KEYPOINT_MASK_SHAPE[0] - 1)
        x_real_map = x_real_map * (1 - x_boundary_bool) + x_boundary_bool * (config.KEYPOINT_MASK_SHAPE[1] - 1)

        valid_loc = tf.logical_and(
            tf.logical_and(x_real_map > 0, x_real_map < config.KEYPOINT_MASK_SHAPE[0]),
            tf.logical_and(y_real_map > 0, y_real_map < config.KEYPOINT_MASK_SHAPE[1])
        )
        valid = tf.logical_and(valid_loc, vis)
        keypoint_weights.append(valid)

        valid = tf.cast(valid, tf.int32)
        x_real_map = x_real_map * tf.cast(valid, tf.int32)
        y_real_map = y_real_map * tf.cast(valid, tf.int32)

        #calculate the keypoint label betwween[0, map_h*map_w)
        keypoint_label = y_real_map * config.KEYPOINT_MASK_SHAPE[1] + x_real_map
        keypoint_label = tf.expand_dims(keypoint_label, -1)
        keypoint_labels.append(keypoint_label)

    # shape:[N_roi, num_keypoint]
    keypoint_labels = tf.cast(tf.concat(keypoint_labels, axis=1), tf.int32)
    keypoint_weights = tf.cast(tf.stack(keypoint_weights, axis=1), tf.int32)

    return keypoint_labels, keypoint_weights

class DetectionKeypointTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,keypoint_weights
    and keypoint_masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals, Here N <= RPN_TRAIN_ANCHORS_PER_IMAGE(256)
               because of the NMS e.t.c
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_keypoints: [batch, MAX_GT_INSTANCES, NUM_KEYPOINTS, 3]
                (x, y, v)
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    keypoint_weights and keypoint_masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinements.
    target_keypoints: [batch, TRAIN_ROIS_PER_IMAGE, NUM_KEYPOINTS)
                 Keypoint labels cropped to bbox boundaries and resized to neural
                 network output size. Maps keypoints from the half-open interval [x1, x2) on continuous image
                coordinates to the closed interval [0, HEATMAP_SIZE - 1]

    target_keypoint_weights: [batch, TRAIN_ROIS_PER_IMAGE, NUM_KEYPOINTS), bool type
                 Keypoint_weights, 0: isn't visible, 1: visilble

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionKeypointTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]  # [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        gt_keypoints = inputs[1]  # [batch, MAX_GT_INSTANCES, NUM_KEYPOINTS, 3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["target_keypoint","target_keypoint_weight"]
        outputs = utils.batch_slice(
            [proposals, gt_keypoints],
            lambda x, y: detection_keypoint_targets_graph(x, y, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs


############################################################
#  Feature Pyramid Network Heads
############################################################

def build_fpn_keypoint_graph(rois, feature_maps,
                         image_shape, pool_size, num_keypoints, ROI_MODE):
    """Builds the computation graph of the keypoint head of Feature Pyramid Network.
    """

    # ROI Pooling
    # Shape: [batch, num_rois, pool_height, pool_width, channels]
    if ROI_MODE == "ROI_MULTI":
        feature_maps[0] = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p2_maxpooling")(feature_maps[0])
        feature_maps[2] = KL.Lambda(lambda z: tf.image.resize_bilinear(z, [64, 64]), name="fpn_p4_unsample")(
            feature_maps[2])
        feature_maps[3] = KL.Lambda(lambda z: tf.image.resize_bilinear(z, [64, 64]), name="fpn_p5_unsample")(
            feature_maps[3])
        all_x = KL.merge([feature_maps[0], feature_maps[1], feature_maps[2], feature_maps[3]], mode="concat",
                         concat_axis=3)
        x = KL.Lambda(lambda z: K.expand_dims(z, 0), name="mrcnn_keypoint_mask_expand_dim")(all_x)
    elif ROI_MODE == "ROI_SINGLE":
        # feature_maps[0] = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p2_maxpooling")(feature_maps[0])
        x = KL.Lambda(lambda z: K.expand_dims(z, 0), name="mrcnn_keypoint_mask_expand_dim")(feature_maps[0])
    elif ROI_MODE == "ROI_ALIGN":
        x = PyramidROIAlign([pool_size, pool_size], image_shape, name="roi_align_keypoint_mask")([rois] + feature_maps)
    else:
        assert "Your ROI_MODE is wrong!"

    for i in range(8):
        x = KL.TimeDistributed(KL.Conv2D(512, (5, 5), padding="same"),
                               name="mrcnn_keypoint_mask_conv{}".format(i + 1))(x)

        # x = KL.TimeDistributed(BatchNorm(axis=3),
        #                        name='mrcnn_keypoint_mask_bn{}'.format(i + 1))(x)
        x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(num_keypoints, (1, 1), padding="same"),
                           name="mrcnn_keypoint_output")(x)
    # x = KL.TimeDistributed(
    #     KL.Lambda(lambda z: tf.image.resize_bilinear(z, [128, 128])), name="mrcnn_keypoint_mask_upsample_1")(x)
    #
    # #shape: batch_size, num_roi, 56, 56, num_keypoint
    # x = KL.TimeDistributed(
    #     KL.Lambda(lambda z: tf.image.resize_bilinear(z, [128, 128])), name="mrcnn_keypoint_mask_upsample_2")(x)
    # shape: batch_size, num_roi, 128, 128, num_keypoint
    # shape: batch_size, num_roi, num_keypoint, 128, 128
    x = KL.TimeDistributed(KL.Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2])), name="mrcnn_keypoint_mask_transpose")(x)
    s = K.int_shape(x)
    x = KL.Reshape((1, num_keypoints, -1), name='mrcnn_keypoint_mask_reshape')(x)
    return x


############################################################
#  Loss Functions
############################################################

def keypoint_mrcnn_mask_loss_graph(target_keypoints, target_keypoint_weights, pred_keypoints_logit, weight_loss = True, mask_shape=[56,56],number_point=15):
    """Mask softmax cross-entropy loss for the keypoint head.

    target_keypoints: [batch, num_rois, num_keypoints].
        A int32 tensor of values between[0, 56*56). Uses zero padding to fill array.
    keypoint_weight:[num_person, num_keypoint]
        0: not visible for the coressponding roi
        1: visible for the coressponding roi
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_keypoints_logit: [batch, proposals, num_keypoints, height*width] float32 tensor
                with values from 0 to 1.
    """

    # Reshape for simplicity. Merge first two dimensions into one.
    #shape:[N]
    # Only positive person ROIs contribute to the loss. And only
    # the people specific mask of each ROI.
    ###Step 1 Get the positive target and predict keypoint masks
        # reshape target_keypoint_weights to [N, num_keypoints]
    target_keypoint_weights = K.reshape(target_keypoint_weights, (-1, number_point))
        # reshape target_keypoint_masks to [N, 17]
    target_keypoints = K.reshape(target_keypoints, (
        -1,  number_point))
    target_keypoints = tf.cast(target_keypoints, tf.int64)
    target_keypoint_weights = tf.cast(target_keypoint_weights, tf.float32)
    # reshape pred_keypoint_masks to [N, 17, 56*56]
    pred_keypoints_logit = K.reshape(pred_keypoints_logit,
                                    (-1, number_point, mask_shape[0]*mask_shape[1]))

    loss = K.switch(tf.size(target_keypoints) > 0,
                    lambda: tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_keypoints_logit, labels=target_keypoints),
                    lambda: tf.constant(0.0))
    loss = loss * target_keypoint_weights

    if(weight_loss):
        loss = K.switch(tf.reduce_sum(target_keypoint_weights) > 0,
                        lambda: tf.reduce_sum(loss) / tf.reduce_sum(target_keypoint_weights),
                        lambda: tf.constant(0.0)
                        )
    else:
        loss = K.mean(loss)
    loss = tf.reshape(loss,[1,1])

    return loss


############################################################
#  Data Generator
############################################################


def load_image_gt_keypoints(dataset, config, image_id):
    """Load and return ground truth data for an image (image, keypoint_mask, keypoint_weight, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks and keypoints that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    keypoints:[num_person, num_keypoint, 3] (x, y, v) v value is as belows:
        0: not visible and without annotations
        1: not visible but with annotations
        2: visible and with annotations
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    image = dataset.load_image(image_id)
    keypoints,  class_ids = dataset.load_keypoints(image_id)
    assert (config.NUM_KEYPOINTS == keypoints.shape[1])

    image, window, scale_xy = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM)

    keypoints = utils.resize_keypoints(keypoints, image.shape[:2], scale_xy)

    return image, window, keypoints

def data_generator_keypoint(dataset, config, shuffle=True, augment=True, batch_size=1):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, keypoint_masks, keypoint_weights, masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_keypoints: [batch, MAX_GT_INSTANCES, NUM_KEYPOINT, 3].
        (x, y, v) and v is in (0, 1,2)
        0: not visible and without annotations
        1: not visible but with annotations
        2: visible and with annotations
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            image, window, gt_keypoints = load_image_gt_keypoints(dataset, config, image_id)

            Num_keypoint = np.shape(gt_keypoints)[1]

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_windows = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)

                batch_gt_keypoints = np.zeros((batch_size, config.MAX_GT_INSTANCES, Num_keypoint, 3))

            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_windows[b, 0] = window
            batch_gt_keypoints[b, :gt_keypoints.shape[0], :, :] = gt_keypoints

            b += 1

            # Batch full?
            # input_image, input_image_meta,
            #  input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_keypoint_masks,
            #  input_gt_keypoint_weigths
            if b >= batch_size:
                inputs = [batch_images, batch_gt_windows, batch_gt_keypoints]
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        # Inputs
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_gt_windows = KL.Input(
            shape=[1, 4], name="input_gt_windows", dtype=tf.float32)

        h, w = K.shape(input_image)[1], K.shape(input_image)[2]
        image_scale = K.cast(K.stack([h, w, h, w], axis=0), tf.float32)
        gt_windows = KL.Lambda(lambda x: x / image_scale, name="gt_windows")(input_gt_windows)

        if mode == "training":
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates

            # Normalize coordinates
            keypoint_scale = K.cast(K.stack([w, h, 1], axis=0), tf.float32)
            input_gt_keypoints = KL.Input(shape=[1, config.NUM_KEYPOINTS, 3])  # [num_image, 1, NUM_KEYPOINTS, 3]
            gt_keypoints = KL.Lambda(lambda x: x / keypoint_scale, name="gt_keypoints")(input_gt_keypoints)

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet101", stage5=True)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(256, (5, 5), padding="SAME", name="fpn_p2")(P2)
        # P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        # P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        # P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        # P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        mrcnn_feature_maps = [P2]

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        rpn_rois = gt_windows

        if mode == "training":
            target_rois = rpn_rois

            target_keypoint, target_keypoint_weight = DetectionKeypointTargetLayer(config, name="proposal_targets") ([target_rois, gt_keypoints])

            keypoint_mrcnn_mask = build_fpn_keypoint_graph(target_rois, mrcnn_feature_maps,
                                              config.IMAGE_SHAPE,
                                              config.KEYPOINT_MASK_POOL_SIZE,
                                              config.NUM_KEYPOINTS,
                                              config.ROI_MODE)

            # Losses
            keypoint_loss = KL.Lambda(lambda x: keypoint_mrcnn_mask_loss_graph(*x, weight_loss=config.WEIGHT_LOSS, number_point=config.NUM_KEYPOINTS,mask_shape=config.KEYPOINT_MASK_SHAPE), name="keypoint_mrcnn_mask_loss")(
                [target_keypoint, target_keypoint_weight, keypoint_mrcnn_mask])

            # Model generated
            # batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, \
            # batch_gt_boxes, batch_gt_keypoint, batch_gt_masks
            inputs = [input_image, input_gt_windows, input_gt_keypoints]

            outputs = [keypoint_loss]

            model = KM.Model(inputs, outputs, name='mask_keypoint_mrcnn')
        else:
            keypoint_mrcnn = build_fpn_keypoint_graph(rpn_rois, mrcnn_feature_maps,
                                                           config.IMAGE_SHAPE,
                                                           config.KEYPOINT_MASK_POOL_SIZE,
                                                           config.NUM_KEYPOINTS,
                                                           config.ROI_MODE)

            #shape: Batch, N_ROI, Number_Keypoint, height*width
            keypoint_mcrcnn_prob = KL.Activation("softmax", name="mrcnn_prob")(keypoint_mrcnn)
            model = KM.Model([input_image, input_gt_windows],
                             [keypoint_mcrcnn_prob],
                             name='keypoint_mask_rcnn')

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["keypoint_mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(
                tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.keras_model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[
                                 None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(
                layer.output, keep_dims=True))

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}_{:%Y%m%dT%H%M}".format(
            self.config.ROI_MODE, now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.ROI_MODE))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data keypoint generators

        train_generator = data_generator_keypoint(train_dataset, self.config, shuffle=True,
                                        batch_size=self.config.BATCH_SIZE, augment = True)
        val_generator = data_generator_keypoint(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE,
                                       augment=False)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = max(self.config.BATCH_SIZE // 2, 2)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            # validation_data=next(val_generator),
            # validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        windows = []
        scales = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale_xy = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM)
            molded_image = mold_image(molded_image, self.config)
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            scales.append(scale_xy)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        windows = np.stack(windows)
        scales = np.stack(scales)
        return molded_images, windows, scales

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, full_masks

    def unmold_keypoint_detections(self, detections, mrcnn_keypoints, image_shape, keypoint_threshold=0.00,
                                   keypoint_mask_shape=[128, 128]):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_keypoints: [N, num_keypoints, height*width]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, N] Instance masks
        keypoints:[N, num_keypoints]
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        N = detections.shape[0]

        # Resize masks to original image size and set boundary threshold.
        keypoints = []

        for i in range(N):
            # Convert neural network mask to full size mask
            keypoint = utils.unmold_keypoint_mask(mrcnn_keypoints[i], image_shape,
                                                  keypoint_mask_shape=keypoint_mask_shape,
                                                  keypoint_threshold=keypoint_threshold)

            keypoints.append(keypoint)

        keypoints = np.stack(keypoints, axis=0) if keypoints else np.empty((0,) + (mrcnn_keypoints.shape[1], 3))

        return keypoints

    def detect_keypoint(self, images, keypoint_mask_shape=[128, 128], augment=True):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.list[batch,(1024,1024,3)]

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [batch, N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [batch, N] int class IDs
        scores: [batch, N] float probability scores for the class IDs
        keypoints: [batch, N, num_keypoints, 3] (x, y, v), keypoint x, y coordinate and valid

        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"
        flag = False
        original_image_shapes = [list(img.shape) for img in images]
        if augment and random.randint(0, 1):
            original_image = cv2.flip(images[0], 1)
            flag = True
        else:
            original_image = images[0]
        # Mold inputs to format expected by the neural network
        molded_images, windows, scales = self.mold_inputs([original_image])

        # Run human pose detection
        detections = np.array([windows])
        # detections = np.array([[[0, 0, 1024, 1024]]])
        # print(detections)
        mrcnn_keypoint_prob = self.keras_model.predict([molded_images, detections], verbose=0)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_keypoints = \
                self.unmold_keypoint_detections(detections[i], mrcnn_keypoint_prob[i], image.shape,
                                                keypoint_mask_shape=keypoint_mask_shape)
            if flag:
                keypoint_flip_map = utils.get_keypoints()
                final_keypoints = utils.flip_keypoints(keypoint_flip_map, final_keypoints, image.shape[1])

            results.append({
                "keypoints": final_keypoints,
            })
        return results

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            # print(name)
            name = re.compile(name.replace("/", r"(\_\d+)*/"))
            # print(name)

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, model_inputs, outputs,TEST_MODE = "training"):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        # print(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Run inference

        # TODO: support training mode?
        if TEST_MODE == "training":
            # model_images = model_inputs[0]
            # image_metas = model_inputs[1]
            # rpn_match = model_inputs[2]
            # rn_bbox = model_inputs[3]
            # gt_class_ids = model_inputs[4]
            # gt_boxes = model_inputs[5]
            # gt_keypoint_masks = model_inputs[6]
            # gt_keypoint_weights = model_inputs[7]
            model_in = model_inputs
            # if not config.USE_RPN_ROIS:
            #     model_in.append(target_rois)
            if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
                model_in.append(1.)
            outputs_np = kf(model_in)
        else:
            images = model_inputs[0]
            molded_images, image_metas, windows = self.mold_inputs(images)
            model_in = [molded_images, image_metas]
            if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
                model_in.append(0.)
            outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL