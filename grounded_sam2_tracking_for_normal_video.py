import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import cv2
import dataclasses
from supervision.draw.color import Color, ColorPalette
import numpy as np
import json
import pandas as pd
import supervision as sv
from datetime import datetime
import torch
import torchvision
import torch.nn.functional as F
from groundingdino.util.inference import Model

from sam2.build_sam import build_sam2, build_sam2_video_predictor_v2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.utils import update_new_masks
from sam2.utils.misc import preprocess_frame
from utils.video_utils import create_video_from_images

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disable torch gradient computation
torch.set_grad_enabled(False)

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = './grounding_dino/config/GroundingDINO_SwinT_OGC.py'
GROUNDING_DINO_CHECKPOINT_PATH = "/home/zhaoyu/checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.5

class GroundedSAMPerception():
    def __init__(
        self,
        classes=['cup'],
        detection_step=1, # each frame has to be detected by default
        use_sam2_tracker=True,
        input_img_size=512, # SAM2 official setting is 1024, but 512 is more efficient with similar performance
    ):
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        )
        # Building MobileSAM predictor
        self.sam2_checkpoint = "/home/zhaoyu/project/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt" # you need to download model, and modify your own path
        self.input_img_size = input_img_size
        if self.input_img_size == 1024:
            self.model_cfg = "sam2_hiera_l.yaml"
        elif self.input_img_size == 512:
            self.model_cfg = "sam2_hiera_l_img512.yaml" # a more efficient setting to inference faster
        else: 
            raise Exception("Sorry, the input image size {} is not exist in current yaml files".format(self.input_img_size))
        sam2_image_model = build_sam2(self.model_cfg, self.sam2_checkpoint)
        self.sam_predictor = SAM2ImagePredictor(sam2_image_model, input_image_size=input_img_size)
        self.use_sam2_tracker = use_sam2_tracker
        # update new object in non-first frame needs to reinitialize a video_predictor  
        self.video_predictor_list, self.inference_state_list = [], []
        self.frame_id = 0
        self.objects_count = 0
        self.object_dict = {}
        self.classes = classes
        self.detection_step = detection_step
        self.detections_from_last_frame = None

    def vis_result_fast(
        self,
        image: np.ndarray, 
        detections: sv.Detections, 
        classes: list[str], 
        # color = ColorPalette.LEGACY, # ROBOFLOW,LEGACY
        color = ColorPalette.default(),
        instance_random_color: bool = True,
        draw_bbox: bool = True,
    ) -> np.ndarray:
        '''
        Annotate the image with the detection results. 
        This is fast but of the same resolution of the input image, thus can be blurry. 
        '''
        # annotate image with detections
        box_annotator = sv.BoxAnnotator(
            color = color,
            text_scale=0.3,
            text_thickness=1,
            text_padding=2,
        )
        mask_annotator = sv.MaskAnnotator(
            color = color
        )

        labels = [
            f"{classes[class_id]} obj {track_id}" 
            for _, _, confidence, class_id, track_id, _ 
            in detections]
        
        if instance_random_color:
            # generate random colors for each segmentation
            # First create a shallow copy of the input detections
            detections = dataclasses.replace(detections)
            detections.class_id = np.arange(len(detections))
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        
        if draw_bbox:
            detections.xyxy[:, 1][detections.xyxy[:, 1]<=0] = 0
            detections.xyxy[:, 1] += 10
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return annotated_image, labels

    def perceptron_img(self, img_rgb):
        torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        # Predict classes and hyper-param for GroundingDINO
        CLASSES = self.classes
        print('==========det class:', CLASSES)
        
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=img_bgr,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        valid_idx = detections.class_id!=None
        detections.xyxy = detections.xyxy[valid_idx]
        detections.confidence = detections.confidence[valid_idx]
        detections.class_id = detections.class_id[valid_idx]
        # convert detections to masks
        detections.mask = self.segment(image=img_bgr, xyxy=detections.xyxy)
        print('det mask:', detections.mask.shape)

        return img_bgr, detections
    
    def video_predictor_init(self, img_bgr, detections):
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        height, width, _ = img_bgr.shape
        video_predictor = build_sam2_video_predictor_v2(self.model_cfg, self.sam2_checkpoint)
        inference_state = video_predictor.init_state(first_frame=preprocess_frame(img_bgr, self.input_img_size), video_width=width, video_height=height)
        # self.objects_count有滞后性，代表之前object的数量
        # for idx, mask in enumerate(detections.mask, start=self.objects_count+1):
        for idx, mask in enumerate(detections.mask):
            # labels = np.ones((1), dtype=np.int32)
            object_id = idx + self.objects_count + 1
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame=preprocess_frame(img_bgr, self.input_img_size),
                frame_idx=self.frame_id,
                obj_id=object_id,
                mask=mask
            )
            self.object_dict[object_id] = detections.class_id[idx]
        return video_predictor, inference_state
    

    def perceptron_video(self, img_rgb):
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out_obj_ids_all, out_mask_logits_all = [], []
        for video_predictor, inference_state in zip(self.video_predictor_list, self.inference_state_list):
            _, out_obj_ids, out_mask_logits = video_predictor.propagate_in_video(inference_state, preprocess_frame(img_bgr, self.input_img_size), self.frame_id)
            out_obj_ids_all.extend(out_obj_ids)
            out_mask_logits_all.append(out_mask_logits)
        out_mask_logits_all = torch.concat(out_mask_logits_all, dim=0)
        out_mask_logits_all = (out_mask_logits_all > 0).cpu().numpy().squeeze(axis=1)
        xyxy = sv.mask_to_xyxy(out_mask_logits_all)
        detections = sv.Detections(xyxy=xyxy)
        detections.mask = out_mask_logits_all
        detections.class_id = np.array([self.object_dict[obj_id] for obj_id in out_obj_ids_all])
        detections.confidence = np.array([1 for _ in out_obj_ids_all]) # 赋个假的
        detections.tracker_id = np.array(out_obj_ids_all)
        if len(self.video_predictor_list) > 1:
            detections = detections.with_nms(threshold=0.8, class_agnostic=False)
        return img_bgr, detections
    
    def perceptron_video_only_update(self, img_rgb, track_detections):
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out_frame_idx, out_obj_ids, out_mask_logits = self.video_predictor_list[-1].propagate_in_video(self.inference_state_list[-1], preprocess_frame(img_bgr, self.input_img_size), self.frame_id)
        out_mask_logits = (out_mask_logits > 0).cpu().numpy().squeeze(axis=1)
        xyxy = sv.mask_to_xyxy(out_mask_logits)
        detections = sv.Detections(xyxy=xyxy)
        detections.mask = out_mask_logits
        detections.class_id = np.array([self.object_dict[obj_id] for obj_id in out_obj_ids])
        detections.confidence = np.array([1 for _ in out_obj_ids]) # 赋个假的
        detections.tracker_id = np.array(out_obj_ids)
        # merge
        merged_detections = sv.Detections.merge([track_detections, detections])
        return img_bgr, merged_detections
    
    def predict(
        self,
        img_rgb,
    ):
        """
        Arguments:
            img_rgb: image of shape (H, W, 3) (in RGB order)
        Returns:
            detetctions
        """
        print('frame_id:', self.frame_id)
        if self.use_sam2_tracker:
            if self.frame_id == 0:
                img_bgr, detections = self.perceptron_img(img_rgb)
                video_predictor, inference_state = self.video_predictor_init(img_bgr, detections)
                self.video_predictor_list.append(video_predictor)
                self.inference_state_list.append(inference_state)
                img_bgr, detections = self.perceptron_video(img_rgb)
                self.detections_from_last_frame = detections
                self.objects_count = len(detections.mask)
            else:
                
                if not self.frame_id % self.detection_step:
                    img_bgr, detections = self.perceptron_img(img_rgb)
                    objects_count, new_detections = update_new_masks(detections, self.detections_from_last_frame, iou_threshold=0.4, objects_count=self.objects_count)
                    if new_detections is not None:
                        # 按照目前的代码逻辑需要重新初始化
                        # self.video_predictor.reset_state(self.inference_state)
                        video_predictor, inference_state = self.video_predictor_init(img_bgr, new_detections)
                        self.objects_count = objects_count
                        self.video_predictor_list.append(video_predictor)
                        self.inference_state_list.append(inference_state)
                        # 可以改成只infer new video predictor，提高效率
                        # img_bgr, detections = self.perceptron_video(img_rgb)
                        # img_bgr, track_detections = self.perceptron_video_only_update(img_rgb, track_detections)
                # else:
                    # print("no new detected object")
                img_bgr, detections = self.perceptron_video(img_rgb)
            # 再转回去，避免影响后面网络的数据类型
            # torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()
        else:
            img_bgr, detections = self.perceptron_img(img_rgb)
        annotated_image, labels = self.vis_result_fast(img_bgr, detections, self.classes)
        self.frame_id += 1
        return detections, annotated_image

    # Prompting SAM with detected boxes
    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """
        Get masks for all detected bounding boxes using SAM
        Arguments:
            image: image of shape (H, W, 3)
            xyxy: bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        Returns:
            masks: masks of shape (N, H, W)
        """
        # 注意：一定要把数据类型转为torch.bfloat16，否则会报错
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        self.sam_predictor.set_image(image)
        # 只运行一次出全部box的mask，根据粗略统计可以提速好几倍 
        result_masks, scores, logits = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=xyxy,
            multimask_output=False,
        )
        if result_masks.ndim == 4:
            result_masks = result_masks.squeeze(1)
        
        # 再转回去，避免影响后面网络的数据类型
        # torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()
        return np.array(result_masks).astype(bool)

if __name__ == '__main__':
    import time
    perception = GroundedSAMPerception(detection_step=3, input_img_size=512, use_sam2_tracker=True)
    INPUT_PATH = '/home/zhaoyu/Downloads/cup4/'
    SAVE_TRACKING_RESULTS_DIR = '/home/zhaoyu/project/Grounded-SAM-2/track_0829/'
    OUTPUT_VIDEO_PATH = '/home/zhaoyu/project/Grounded-SAM-2/cup_0829.mp4'
    os.makedirs(SAVE_TRACKING_RESULTS_DIR, exist_ok=True)
    frame_names = [
        p for p in os.listdir(INPUT_PATH)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    for i in range(len(frame_names)):
        img = cv2.imread(INPUT_PATH+frame_names[i])[:, :, -1::-1]
        tic = time.time()
        detections, annotated_image = perception.predict(img)
        cv2.imwrite(SAVE_TRACKING_RESULTS_DIR+'{:05d}.jpg'.format(i), annotated_image)
        print('cost time:', time.time()-tic)
    create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH, frame_rate=2)