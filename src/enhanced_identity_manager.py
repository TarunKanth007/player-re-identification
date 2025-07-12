# File: src/enhanced_identity_manager.py

import numpy as np
from scipy.spatial.distance import cosine

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def center_distance(box1, box2):
    c1 = bbox_center(box1)
    c2 = bbox_center(box2)
    return np.linalg.norm(np.array(c1) - np.array(c2))

def cosine_similarity(f1, f2):
    return 1 - cosine(f1, f2)

class EnhancedIdentityManager:
    def __init__(self, sim_threshold=0.85, max_inactive=30, max_center_dist=100):
        self.sim_threshold = sim_threshold
        self.max_inactive = max_inactive  # max allowed frames of inactivity
        self.max_center_dist = max_center_dist
        self.next_id = 1
        self.tracks = {}  # player_id -> {'bbox': ..., 'feature': ..., 'last_seen': frame_id}

    def assign_ids_single_camera(self, bboxes, features, frame_id):
        assigned_ids = {}
        unmatched = set(range(len(bboxes)))

        # Step 1: Try to match new detections with existing tracks
        for track_id, track in self.tracks.items():
            best_idx = -1
            best_score = -1

            for i in unmatched:
                sim = cosine_similarity(track['feature'], features[i])
                dist = center_distance(track['bbox'], bboxes[i])

                if sim > self.sim_threshold and dist < self.max_center_dist:
                    if sim > best_score:
                        best_score = sim
                        best_idx = i

            if best_idx >= 0:
                assigned_ids[best_idx] = track_id
                self.tracks[track_id] = {
                    'bbox': bboxes[best_idx],
                    'feature': features[best_idx],
                    'last_seen': frame_id
                }
                unmatched.remove(best_idx)

        # Step 2: Assign new IDs to unmatched detections
        for i in unmatched:
            assigned_ids[i] = self.next_id
            self.tracks[self.next_id] = {
                'bbox': bboxes[i],
                'feature': features[i],
                'last_seen': frame_id
            }
            self.next_id += 1

        # Step 3: Remove stale tracks
        to_remove = []
        for track_id, track in self.tracks.items():
            if frame_id - track['last_seen'] > self.max_inactive:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]

        # Output mapping from bbox index to assigned player ID
        return {i: assigned_ids[i] for i in range(len(bboxes)) if i in assigned_ids}
