from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="dinosaur-pose.yaml", epochs=100, imgsz=640)

metrics = model.val()
