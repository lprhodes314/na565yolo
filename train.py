from ultralytics import YOLO

# Load a pretrained model. Yolo has nano (n), small(s), medium(m), large(l), extra large (x) pretrained models.
model = YOLO("yolo11n.pt")

# Train the model with MPS. Update options as needed.
results = model.train(data="datasets/kitti.yaml", epochs=100, imgsz=640, device=0, save=True, patience=30, workers=1)

# TO RESUME INTERRUPTED TRAINING
# Load a partially trained model.
# model = YOLO("udacity_n_partial.pt")
# 
# Finish training. It will remember training settings).
# results = model.train(resume=True)