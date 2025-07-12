# File: src/tracker.py
import numpy as np
from scipy.spatial.distance import cosine
from src.bms import BMSSignature


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area else 0


def center_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return np.linalg.norm([cx1 - cx2, cy1 - cy2])


class BMSTracker:
    def __init__(self, iou_thresh=0.2, cos_thresh=0.6, dist_thresh=70, lock_frames=15, max_age=60):
        self.tracks = {}  # {id: {bbox, feature, age, last_seen_frame}}
        self.bms_map = {}
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.cos_thresh = cos_thresh
        self.dist_thresh = dist_thresh
        self.lock_frames = lock_frames
        self.max_age = max_age

    def match(self, bbox, feat, frame_idx):
        best_id = None
        best_score = float('inf')

        for tid, track in self.tracks.items():
            prev_box = track['bbox']
            prev_feat = track['feature']
            last_seen = track['last_seen_frame']
            if frame_idx - last_seen > self.max_age:
                continue

            iou_score = iou(bbox, prev_box)
            cos_sim = cosine(feat, prev_feat)
            center_shift = center_distance(bbox, prev_box)
            bms_dist = BMSSignature.compare(BMSSignature.compute(bbox, frame_idx), self.bms_map.get(tid, {}))

            if iou_score > self.iou_thresh and cos_sim < self.cos_thresh and center_shift < self.dist_thresh:
                score = cos_sim + (1 - iou_score) + center_shift / 100 + bms_dist
                if score < best_score:
                    best_score = score
                    best_id = tid

        return best_id

    def update(self, detections, features, frame_idx):
        updated_tracks = []
        matched_ids = set()
        new_assignments = []

        for det, feat in zip(detections, features):
            bbox = det[:4]
            best_id = self.match(bbox, feat, frame_idx)

            if best_id is not None and best_id not in matched_ids:
                self.tracks[best_id].update({
                    'bbox': bbox,
                    'feature': feat,
                    'age': 0,
                    'last_seen_frame': frame_idx
                })
                self.bms_map[best_id] = BMSSignature.compute(bbox, frame_idx)
                updated_tracks.append({'id': best_id, 'bbox': bbox})
                matched_ids.add(best_id)
            else:
                new_assignments.append((bbox, feat))

        # Add new tracks
        for bbox, feat in new_assignments:
            new_id = self.next_id
            self.tracks[new_id] = {
                'bbox': bbox,
                'feature': feat,
                'age': 0,
                'last_seen_frame': frame_idx
            }
            self.bms_map[new_id] = BMSSignature.compute(bbox, frame_idx)
            updated_tracks.append({'id': new_id, 'bbox': bbox})
            matched_ids.add(new_id)
            self.next_id += 1

        # Update ages of unmatched tracks
        expired_ids = []
        for tid, track in self.tracks.items():
            if tid not in matched_ids:
                track['age'] += 1
                if track['age'] > self.max_age:
                    expired_ids.append(tid)

        for tid in expired_ids:
            del self.tracks[tid]
            self.bms_map.pop(tid, None)

        return updated_tracks