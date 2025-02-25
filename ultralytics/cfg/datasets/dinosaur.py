from ultralytics import YOLO

import cv2
# Load a model
# model = YOLO("yolo11n.pt")
model = YOLO("best.pt")


def train():
    # Train the model
    train_results = model.train(
        data="dinosaur.yaml",  # path to dataset YAML
        epochs=200,  # number of training epochs
        imgsz=640,  # training image size
        device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        workers=6,
    )


def validation():
    # Evaluate model performance on the validation set
    metrics = model.val()

def parse():
    # Perform object detection on an image
    path = "D:/work/self/ultralytics/dinosaur.png"
    # image = cv2.imread(path)
    # cropped_frame = image[0:456, 220:817]
    results = model(path)
    # print(results)
    tuili_jieguo = []
    for result in results:
        # print(result)
        if len(result.boxes.cls) > 0:
            for i in range(len(result.boxes.cls)):
                # 获服类别
                leibie_id = int(result.boxes.cls[i].item())  # 获取类测id
                leibie = result.names[leibie_id]  # 获取类列的名称
                # 获取相似度
                xiangsidu = str(result.boxes.conf[i].item())[0:4]
                # 获取坐标值,2个，左上角和右上角
                zuobiao = result.boxes.xyxy[i].tolist()
                # 存入列表中
                obj = {'类别': leibie, '相似度': xiangsidu, '坐标': zuobiao}
                tuili_jieguo.append(obj)

                print(obj)
    # results[0].show()

def export_ncnn():
    #Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model
    print("path ",path)

if __name__ == '__main__':
    # train()
    # validation()
    parse()
    # export_ncnn()


