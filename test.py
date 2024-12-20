from ultralytics import YOLO

# Load the trained model for evaluation
model = YOLO("kitti_m_98.pt")

# Use the associated test data for evaluation
metrics = model.val(data="datasets/kitti_eval.yaml")
print(metrics.box.map)  # mAP50-95