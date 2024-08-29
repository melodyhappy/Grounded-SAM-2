import numpy as np
import supervision as sv

def update_new_masks(det_results, track_results, iou_threshold=0.8, objects_count=0):
    new_mask, new_xyxy, new_class_id, new_confidence, new_tracker_id = [], [], [], [], []
    for det_idx, det_mask  in enumerate(det_results.mask):
        flag = 0 
        if det_mask.sum() == 0:
            continue
        for track_idx, track_mask  in enumerate(track_results.mask):
            iou = calculate_iou(det_mask, track_mask)
            if iou > iou_threshold:
                flag = track_results.tracker_id[track_idx]
                break
        if not flag:
            # 出现新物体
            objects_count += 1
            flag = objects_count
            new_mask.append(det_mask)
            new_xyxy.append(det_results.xyxy[det_idx])
            new_class_id.append(det_results.class_id[det_idx])
            new_confidence.append(det_results.confidence[det_idx])
            new_tracker_id.append(np.array(objects_count))
    if len(new_mask) == 0:
        return objects_count, None        
    new_track_results = sv.Detections(xyxy=np.array(new_xyxy))
    new_track_results.mask = np.array(new_mask)
    new_track_results.class_id = np.array(new_class_id)
    new_track_results.confidence = np.array(new_confidence)
    new_track_results.tracker_id = np.array(new_tracker_id)      
    return objects_count, new_track_results

def calculate_iou(mask1, mask2):
        # Convert masks to float tensors for calculations
        mask1 = mask1.astype(np.float32)
        mask2 = mask2.astype(np.float32)
        
        # Calculate intersection and union
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
        
        # Calculate IoU
        iou = intersection / union
        return iou

