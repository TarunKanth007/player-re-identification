# File: main_cross_camera.py

import cv2
import os
import numpy as np
import time
import pickle
from ultralytics import YOLO
import supervision as sv
from utils import get_center_of_bbox, get_bbox_width
from src.enhanced_feature_extractor import EnhancedFeatureExtractor
from src.enhanced_identity_manager import EnhancedIdentityManager
from scipy.spatial.distance import cosine

SIM_THRESHOLD = 0.85

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i: i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_player_bboxes_and_features(self, frames, extractor):
        detections = self.detect_frames(frames)
        all_bboxes = []
        all_features = []
        all_frames = []

        for frame, detection in zip(frames, detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            for idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[idx] = cls_names_inv["player"]

            frame_bboxes = []
            frame_features = []

            for det in detection_supervision:
                bbox = det[0].astype(int).tolist()
                cls_id = det[3]
                if cls_id == cls_names_inv['player']:
                    feature = extractor.extract(frame, bbox)
                    if feature is not None:
                        frame_bboxes.append(bbox)
                        frame_features.append(feature)

            all_bboxes.append(frame_bboxes)
            all_features.append(frame_features)
            all_frames.append(frame)

        return all_frames, all_bboxes, all_features

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            label = f"ID:{track_id}"
            cv2.putText(
                frame,
                label,
                (x_center - 10, y2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return frame

    def draw_annotations(self, frames, bboxes, id_maps):
        output_frames = []
        for frame, boxes, ids in zip(frames, bboxes, id_maps):
            annotated = frame.copy()
            for i, bbox in enumerate(boxes):
                track_id = ids.get(i)
                annotated = self.draw_ellipse(annotated, bbox, (0, 0, 255), track_id)
            output_frames.append(annotated)
        return output_frames

def assign_ids_cross_camera(self, bboxes, features, frame_id):
    assigned_ids = {}
    unmatched = set(range(len(bboxes)))
    memory_used = set()

    for i in list(unmatched):
        best_id = None
        best_score = -1
        best_dist = float('inf')

        for mem_id, mem in self.tracks.items():
            if mem_id in memory_used:
                continue

            sim = 1 - cosine(mem['feature'], features[i])
            dist = np.linalg.norm(np.array(get_center_of_bbox(mem['bbox'])) - np.array(get_center_of_bbox(bboxes[i])))

            if sim > self.sim_threshold and dist < self.max_center_dist:
                if sim > best_score or (sim == best_score and dist < best_dist):
                    best_score = sim
                    best_dist = dist
                    best_id = mem_id

        if best_id is not None:
            assigned_ids[i] = best_id
            memory_used.add(best_id)
            self.tracks[best_id] = {
                'bbox': bboxes[i],
                'feature': features[i],
                'last_seen': frame_id
            }
            unmatched.remove(i)

    for i in unmatched:
        assigned_ids[i] = self.next_id
        self.tracks[self.next_id] = {
            'bbox': bboxes[i],
            'feature': features[i],
            'last_seen': frame_id
        }
        self.next_id += 1

    return {i: assigned_ids[i] for i in range(len(bboxes)) if i in assigned_ids}

EnhancedIdentityManager.assign_ids_cross_camera = assign_ids_cross_camera

def main_cross_camera():
    model_path = 'models/yolo_player_detector.pt'
    video_t_path = 'data/tacticam.mp4'
    video_b_path = 'data/broadcast.mp4'
    output_t_path = 'outputs/annotated_tacticam.mp4'
    output_b_path = 'outputs/annotated_broadcast.mp4'

    extractor = EnhancedFeatureExtractor()
    detector = Tracker(model_path)
    identity_manager_tacticam = EnhancedIdentityManager(sim_threshold=SIM_THRESHOLD)

    cap_t = cv2.VideoCapture(video_t_path)
    fps = cap_t.get(cv2.CAP_PROP_FPS)
    width = int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_t = []
    while True:
        ret, frame = cap_t.read()
        if not ret:
            break
        frames_t.append(frame)
    cap_t.release()

    print("[INFO] Processing Tacticam frames...")
    frames_t, bboxes_t, feats_t = detector.get_player_bboxes_and_features(frames_t, extractor)

    print("[INFO] Assigning IDs for Tacticam...")
    id_maps_t = []
    for fid, (b, f) in enumerate(zip(bboxes_t, feats_t)):
        ids = identity_manager_tacticam.assign_ids_single_camera(b, f, frame_id=fid)
        id_maps_t.append(ids)

    print("[INFO] Annotating and saving Tacticam output...")
    out_t = cv2.VideoWriter(output_t_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    annotated_t = detector.draw_annotations(frames_t, bboxes_t, id_maps_t)
    for f in annotated_t:
        out_t.write(f)
    out_t.release()

    player_memory = identity_manager_tacticam.tracks.copy()

    cap_b = cv2.VideoCapture(video_b_path)
    frames_b = []
    while True:
        ret, frame = cap_b.read()
        if not ret:
            break
        frames_b.append(frame)
    cap_b.release()

    print("[INFO] Processing Broadcast frames...")
    frames_b, bboxes_b, feats_b = detector.get_player_bboxes_and_features(frames_b, extractor)

    print("[INFO] Matching IDs in Broadcast using cross-camera memory...")
    identity_manager_broadcast = EnhancedIdentityManager(sim_threshold=SIM_THRESHOLD)
    identity_manager_broadcast.tracks = player_memory
    identity_manager_broadcast.next_id = max(player_memory.keys()) + 1

    id_maps_b = []
    for fid, (b, f) in enumerate(zip(bboxes_b, feats_b)):
        ids = identity_manager_broadcast.assign_ids_cross_camera(b, f, frame_id=fid)
        id_maps_b.append(ids)

    print("[INFO] Annotating and saving Broadcast output...")
    out_b = cv2.VideoWriter(output_b_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    annotated_b = detector.draw_annotations(frames_b, bboxes_b, id_maps_b)
    for f in annotated_b:
        out_b.write(f)
    out_b.release()

    print(f"[INFO] Done. Videos saved at: {output_t_path}, {output_b_path}")

if __name__ == '__main__':
    main_cross_camera()
