from selective import resize
from non_max_suppression import nms

import torch
import torchvision.models as models
import torch.nn as nn

import cv2

import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw

test = 'images/000020.jpg'
classes = ["bicycle", "car", "cat", "chair", "dog", "horse", "person"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device, torch.cuda.get_device_name(device))

image = cv2.imread(test)

## test 이미지로부터 영역 추출
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
ssresults = ss.process()

candidate_images = []
candidate_boxes = []

for e, candidate in enumerate(ssresults):
    if e > 2000:
        break

    x, y, w, h = candidate

    img = resize(image, x, y, w, h)

    candidate_images.append(img)
    candidate_boxes.append(candidate)

## 훈련한 모델 로딩

model = models.alexnet()
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 8)
model = model.to(device)

model.load_state_dict(torch.load('runscheckpoint15.pt')["model_state_dict"])
model.eval()

## 훈련한 모델로 예측

candidate_predict = []
candidate_score = []

with torch.no_grad():
    for image in candidate_images:
        image = torch.FloatTensor(image).to(device)

        output = model(image[None, ...].permute(0, 3, 1, 2)) # 배치가 없으므로 None을 가장 첫 인자로 준다
        predict_class = torch.argmax(output).to(device).item()
        candidate_predict.append(predict_class)
        candidate_score.append(torch.softmax(output[0], dim=0)[predict_class].item())

## 그림으로 그리기

temp = Image.open(test).convert('RGB')
draw = ImageDraw.Draw(temp)

candidate_nms = nms(candidate_boxes, candidate_score, candidate_predict)

for e, yn in enumerate(candidate_nms):
    if not yn:
        candidate_predict[e] = 0

for e, predict in enumerate(candidate_predict):

    if predict == 0:
        continue

    else:
        if candidate_score[e] >= 0.998:
            x, y, w, h = candidate_boxes[e]
            draw.rectangle(((x, y), (x + w, y + h)), outline='red')
            draw.text((x, y), str(classes[predict-1]))

plt.figure(figsize=(20, 20))
plt.imshow(temp)
plt.show()