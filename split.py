import supervision as sv

# Splits dataset randomly into train, val, and test subsets.
# Update directories as needed. 

# Load your dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path="datasets/kitti/images",
    annotations_directory_path="datasets/kitti/labels_yolo",
    data_yaml_path="datasets/kitti.yaml"
)

ds_train, ds = dataset.split(split_ratio=0.8, shuffle=True)
ds_val, ds_test = ds.split(split_ratio=0.5, shuffle=True)

ds_val.as_yolo(
    images_directory_path="datasets/kitti/val/images",
    annotations_directory_path="datasets/kitti/val/labels",
    data_yaml_path='datasets/kitti/val.yaml'
)
print("val set saved")

ds_test.as_yolo(
    images_directory_path="datasets/kitti/test/images",
    annotations_directory_path="datasets/kitti/test/labels",
    data_yaml_path='datasets/kitti/test.yaml'
)
print("test set saved")

ds_train.as_yolo(
    images_directory_path="datasets/kitti/train/images",
    annotations_directory_path="datasets/kitti/train/labels",
    data_yaml_path='datasets/kitti/train.yaml'
)
print("train set saved")