# File: src/enhanced_feature_extractor.py
import cv2
import numpy as np
from src.features import FeatureExtractor

class EnhancedFeatureExtractor(FeatureExtractor):
    def extract(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or x2 - x1 < 10 or y2 - y1 < 10:
            return None

        # Resize and convert to HSV
        crop_resized = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

        # Color histogram (H, S channels)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist = np.concatenate([hist_h, hist_s])
        hist = cv2.normalize(hist, hist).flatten()

        # Mean and std color per channel
        mean_color = crop_resized.mean(axis=(0, 1))
        std_color = crop_resized.std(axis=(0, 1))
        color_stats = np.concatenate([mean_color, std_color])

        # Concatenate histogram + stats for stronger feature
        feature_vector = np.concatenate([hist, color_stats])
        return feature_vector
