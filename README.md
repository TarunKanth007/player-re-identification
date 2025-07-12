# Player Re-Identification in Multi-Camera Sports Feeds 🏟️

This project performs consistent player identification across two synchronized football match videos — one from a broadcast view and another from a tacticam (top-view). It uses object detection, handcrafted visual features, and identity memory transfer to assign the same ID to the same player across views.

## 🎯 Objective

- Detect players and assign consistent IDs across videos shot from different angles.
- Improve player tracking accuracy using color histograms and statistical features.
- Generate annotated videos showing detected players with IDs.

## 🔧 Components

- A fine-tuned YOLO model for detecting players.
- Feature extractor that uses HSV histograms and color statistics.
- Identity manager that assigns and tracks consistent player IDs.
- A tracker that detects and draws players with unique IDs.
- Two main scripts:
  - `main.py` processes single-view (tacticam) player tracking.
  - `main_cross_camera.py` matches IDs between tacticam and broadcast videos.

## 🛠️ Setup Instructions

1. Create a virtual environment.
2. Install required packages from `requirements.txt`.
3. Place your input videos `broadcast.mp4` and `tacticam.mp4` in the `data/` folder.
4. Name the model name to `yolo_player_detector`.
5. Place the YOLO model `yolo_player_detector.pt` in the `models/` folder.

## ▶️ Running the Project

To process only the tacticam view and assign IDs:
- Run the `main.py` script.

To perform cross-camera player ID matching between tacticam and broadcast:
- Run the `main_cross_camera.py` script.

The processed videos will be saved in the `outputs/` directory as:
- `annotated_tacticam.mp4`
- `annotated_broadcast.mp4`

## 💡 Core Features

- Assigns same IDs to players across both video feeds.
- Uses color-based visual features for robust player matching.
- Handles player re-identification even with camera angle shifts.
- Visual annotations: Ellipses and IDs drawn over players.

## 🔬 Techniques Used

- YOLO-based detection
- HSV histogram and color statistics as features
- Cosine similarity + center distance for matching
- ByteTrack for tracking
- Cross-camera ID memory reuse

## 🔄 Future Improvements

- Use optical flow to improve temporal consistency
- Apply camera calibration or homography to normalize views
- Use deep learning-based re-ID embeddings (e.g., OSNet, DeepSort+ReID)
- Add GUI or Streamlit interface for interactive inspection

## 📍 Output

Both `annotated_tacticam.mp4` and `annotated_broadcast.mp4` will be saved in the `outputs/` folder with player IDs consistently shown across views.

## 👤 Author

Developed by [Your Name or GitHub Handle]

## 🏁 License

MIT License – free to use and modify.
