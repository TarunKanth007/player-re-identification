# Folder: src/detect.py
from ultralytics import YOLO
import numpy as np

class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.PLAYER_CLASS = 2  # ✅ Player class

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def center_distance(self, boxA, boxB):
        cx1 = (boxA[0] + boxA[2]) / 2
        cy1 = (boxA[1] + boxA[3]) / 2
        cx2 = (boxB[0] + boxB[2]) / 2
        cy2 = (boxB[1] + boxB[3]) / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    def merge_overlapping(self, boxes, iou_thresh=0.5, center_thresh=30):
        merged = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue

            x1, y1, x2, y2, conf = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue

                iou = self.iou((x1, y1, x2, y2), (boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]))
                center_dist = self.center_distance((x1, y1, x2, y2), (boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]))

                if iou > iou_thresh and center_dist < center_thresh:
                    x1 = min(x1, boxes[j][0])
                    y1 = min(y1, boxes[j][1])
                    x2 = max(x2, boxes[j][2])
                    y2 = max(y2, boxes[j][3])
                    conf = max(conf, boxes[j][4])
                    used[j] = True

            merged.append((int(x1), int(y1), int(x2), int(y2), conf))
            used[i] = True

        return merged

    def detect(self, frame):
        results = self.model(frame)[0]
        raw_boxes = []

        for result in results.boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = float(result.conf)
            cls = int(result.cls)

            print(f"[LOG] Detected cls={cls}, conf={conf:.2f}, bbox=({x1},{y1},{x2},{y2})")

            if cls == self.PLAYER_CLASS and conf > 0.4:
                raw_boxes.append((x1, y1, x2, y2, conf))

        merged_boxes = self.merge_overlapping(raw_boxes)

        print(f"[DEBUG] Final merged player detections: {len(merged_boxes)}")
        return merged_boxes
