import time

from ultralytics import YOLO

import cv2
# Load a model
# model = YOLO("yolo11n.pt")
model = YOLO("best.pt")


def train():
    # Train the model
    train_results = model.train(
        data="coco8-pose.yaml",  # path to dataset YAML
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
    # path = "D:/work/self/datasets/coco8-pose/images/val/000000000110.jpg"
    path = "D:/work/self/ultralytics/ultralytics/cfg/datasets/dinosaur_pose/train/images/0fcb4e7b-000000000113.jpg"
    # image = cv2.imread(path)
    # cropped_frame = image[0:456, 220:817]
    results = model(path)

    image = cv2.imread(path)


    # print(results)
    tuili_jieguo = []

    text_index = 0
    for result in results:
        # print(result)

            # cv2.putText(img, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



        # if len(result.keypoints.data) > 0:
        #     for i in range(len(result.keypoints.data)):
        #         conf_len = len(result.keypoints.conf[i])
        #         for i_conf in range(conf_len):
        #             xiangsidu = result.keypoints.conf[i][i_conf].item()
        #
        #             pos = result.keypoints.xy[i][i_conf]
        #             print("相识度",xiangsidu,"坐标",pos)
        #             if xiangsidu > 0.8:
        #                 cv2.putText(image,str(text_index),(int(pos[0]),int(pos[1])),cv2.FONT_HERSHEY_COMPLEX,1, (0, 128, 0), 2, cv2.LINE_AA)
        #                 text_index += 1

        if result.keypoints and result.keypoints.data is not None and len(result.keypoints.data) > 0:
            keypoint_conf_len = len(result.keypoints.data)
            for keypoint_i_conf in range(keypoint_conf_len):
                keypoint_si = 0.91  # result.keypoints.conf[i][keypoint_i_conf].item()
                keypoint_pos = result.keypoints.xy[keypoint_i_conf][0]
                if keypoint_si > 0.9:
                    print("相识度", keypoint_si, "坐标", keypoint_pos)
                    cv2.putText(image, str(text_index), (int(keypoint_pos[0]), int(keypoint_pos[1])),
                                cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 128, 0), 2, cv2.LINE_AA)
                    text_index += 1


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



    results[0].show()
    cv2.imshow('frame', image)
    cv2.waitKey(10)

def export_ncnn():
    #Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model
    print("path ",path)

if __name__ == '__main__':
    # train()
    # validation()
    parse()
    # export_ncnn

    time.sleep(999)


