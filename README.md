# Tracking Tourist Activities and Place Behavior via Computer Vision and Video Data

This GitHub repository accompanies the manuscript *Tracking Tourist Activities and Place Behavior via Computer Vision and Video Data*. This study explores the spatiotemporal behavioral patterns of tourists in micro-level environments to support intelligent management and service improvements, such as optimizing destination facilities and designing routes, ultimately enhancing visitor experiences and destination reputation.

## 1. Project Structure and File Descriptions

- **autofit_sliced_detection.py**: Generates different slicing protocols and determines the optimal slicing protocol based on object detection accuracy evaluations.
- **auto_sliced_mot.py**: Generates visitor motion trajectories using adaptive slicing; stores trace results in a format similar to MOT16.
- **spatialMapping.py**: Maps pixel coordinates in surveillance images to corresponding geographic coordinates. The spatial reference is determined by the `tfw` file from image registration.
- **personal_trace.py**: Converts multi-object tracking data for each frame into trajectory data identified by personal ID, including both image and geographic coordinate systems.
- **visit_pattern_route.py**: Implements recognition of spatiotemporal movement patterns of visitor groups.
- **visit_pattern_stay.py**: Implements recognition of spatiotemporal stay patterns for visitor groups.

## 2. Basic Requirements

The code has been tested on Windows 11 with an RTX 4070TI 12GB GPU. Since this project processes video stream data, GPU capability is required to ensure efficient performance.

### Key Environment and Version Requirements

- **CUDA**: 11
- **torch**: 2.3.0 (`pip install torch==2.3.0`)
- **torchvision**: 0.18.0
- **boxmot**: 10.0.72
- **GDAL**: 3.6.2
- **geopandas**: 0.13.2
- **opencv-python**: 4.8.1.78

## 3. How to use

Run the following scripts in sequence to reproduce the analysis:

```bash
# Determine the slicing protocol for each surveillance scene
python autofit_sliced_detection.py

# Perform multi-object tracking based on the slicing protocol and generate object motion trajectories in image space
python auto_sliced_mot.py

# Convert object trajectories to spatiotemporal trajectories in geographic space
python personal_trace.py

# Extract spatiotemporal route patterns of tourists
python visit_pattern_route.py

# Extract spatiotemporal stay patterns of tourists
python visit_pattern_stay.py
```

## 4. Additional Resources
- YOLO Detection Model: YOLOv8 model weights download from: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt
- Appearance Re-Identification Model: weights/osnet_x1_0_msmt17.pt
