from ultralytics import YOLO

import time

# Load a model
# model = YOLO("yolo11n.pt")
#
model = YOLO("best.pt")


def train():
    # Train the model
    train_results = model.train(
        data="coco8.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )
def parse():
    # Perform object detection on an image
    path = "D:/works/datasets/coco8/images/train/000000000009.jpg"
    # image = cv2.imread(path)
    # cropped_frame = image[0:456, 220:817]
    results = model(path)
    # print(results)
    res = []
    for result in results:
        # print(result)
        if len(result.boxes.cls) > 0:
            for i in range(len(result.boxes.cls)):
                # 获服类别
                clz = int(result.boxes.cls[i].item())  # 获取类测id
                clz_name = result.names[clz]  # 获取类列的名称
                # 获取相似度
                similar = str(result.boxes.conf[i].item())[0:4]
                # 获取坐标值,2个，左上角和右上角
                position = result.boxes.xyxy[i].tolist()
                # 存入列表中
                obj = {'类别': clz_name, '相似度': similar, '坐标': position}
                res.append(obj)

                print(obj)
    # results[0].show()
if __name__ == '__main__':
    # train()

    while True:
        parse()
        # time.sleep(0.1)
    '''
    rtx 4050 Laptop
    # for CUDA 11.8
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
    
    numpy 1.24.0
    '''
    #numpy 1.24.0
    # import torch
    # print(torch.__version__)
    # print(torch.cuda.is_available())
