from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt")

# Export the model to NCNN format
# model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'

# Load the exported NCNN model
# ncnn_model = YOLO("./best.onnx")

# Run inference
results = model("D:/work/self/ultralytics/dnf.png")

results[0].show()