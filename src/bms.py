import numpy as np

class BMSSignature:
    @staticmethod
    def compute(bbox, frame_idx):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        zone = int(cx // 100)
        return {
            'cx': cx,
            'cy': cy,
            'speed': (w * h) / 1000.0,
            'zone': zone,
            'frame': frame_idx
        }

    @staticmethod
    def compare(bms1, bms2):
        dx = abs(bms1['cx'] - bms2['cx'])
        dy = abs(bms1['cy'] - bms2['cy'])
        d_speed = abs(bms1['speed'] - bms2['speed'])
        d_zone = abs(bms1['zone'] - bms2['zone'])
        return (dx + dy + d_speed + d_zone) / 4
