model_path="/home/leekayn/.cache/huggingface/hub/models--JunkyByte--easy_ViTPose/snapshots/2757e82adcccda02f9f7fef66e5a115b7be439fe/torch/wholebody/vitpose-b-wholebody.pth"
yolo_path="/home/leekayn/.cache/huggingface/hub/models--JunkyByte--easy_ViTPose/snapshots/2757e82adcccda02f9f7fef66e5a115b7be439fe/yolov8/yolov8s.pt"
# !pip install huggingface_hub
# MODEL_SIZE = 's'  #@param ['s', 'b', 'l', 'h']
# YOLO_SIZE = 's'  #@param ['s', 'n']
# DATASET = 'wholebody'  #@param ['coco_25', 'coco', 'wholebody', 'mpii', 'aic', 'ap10k', 'apt36k']
# ext = '.pth'
# ext_yolo = '.pt'
#
#
# import os
# from huggingface_hub import hf_hub_download
# MODEL_TYPE = "torch"
# YOLO_TYPE = "torch"
# REPO_ID = 'JunkyByte/easy_ViTPose'
# FILENAME = os.path.join(MODEL_TYPE, f'{DATASET}/vitpose-' + MODEL_SIZE + f'-{DATASET}') + ext
# FILENAME_YOLO = 'yolov8/yolov8' + YOLO_SIZE + ext_yolo
#
# print(f'Downloading model {REPO_ID}/{FILENAME}')
# model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
# yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)

MODEL_SIZE = 'b'
DATASET = 'wholebody'
from easy_ViTPose import VitInference
model = VitInference(model_path, yolo_path, MODEL_SIZE,
                     dataset=DATASET, yolo_size=320, is_video=False)

import cv2
import numpy as np
from PIL import Image

# Đường dẫn tới file ảnh trên máy của bạn
file_path = 'ronaldo1.jpg'
image=cv2.imread(file_path)
# Load ảnh từ file
img = np.array(Image.open(file_path), dtype=np.uint8)
# Tiếp tục với quá trình xử lý ảnh và inference như bạn đã làm
frame_keypoints = model.inference(img)
img,bboxes = model.draw()
# Tìm điểm tâm của các bounding box
bboxes_centers = []
for bbox in bboxes:
  x1, y1, x2, y2 = bbox
  center_x = (x1 + x2) / 2
  center_y = (y1 + y2) / 2
  bboxes_centers.append((center_x, center_y))

# Tính khoảng cách từ điểm tâm đến các cạnh của khung hình
distances_to_edges = []
for center in bboxes_centers:
  distance_to_left = center[0]
  distance_to_right = img.shape[1] - center[0]
  distance_to_top = center[1]
  distance_to_bottom = img.shape[0] - center[1]
  distances_to_edges.append([distance_to_left, distance_to_right, distance_to_top, distance_to_bottom])

# Tìm bounding box có điểm tâm gần khung hình nhất
min_distance_index = np.argmin(np.sum(distances_to_edges, axis=1))

# Cắt ảnh dựa trên bounding box sát khung hình nhất
cropped_image = image[y1:y2, x1:x2]

# Hiển thị ảnh đã cắt
cv2.imshow("Cropped Image", cropped_image)

# Hiển thị hình ảnh bằng OpenCV
cv2.imshow('Image', img[..., ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()