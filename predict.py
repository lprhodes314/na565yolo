from ultralytics import YOLO

# Load the trained model for prediction
model = YOLO("yolo11l.pt")

# Use the model to predict. Many allowed formats. See https://docs.ultralytics.com/modes/predict/
results = model.predict("datasets/udacity/test/images", save=True)