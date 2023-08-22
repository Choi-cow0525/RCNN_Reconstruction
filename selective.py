import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
#            "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
#            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes = ["bicycle", "car", "cat", "chair", "dog", "horse", "person"]

def iou(bb1, bb2):
    ## 오른쪽 좌표가 왼쪽 좌표보다 커야 하고, 위 좌표가 아래 좌표보다 커야 함 그렇지 않을 경우 asserterror
    assert bb1['xmin'] < bb1['xmax']
    assert bb1['ymin'] < bb1['ymax']
    assert bb2['xmin'] < bb2['xmax']
    assert bb2['ymin'] < bb2['ymax']

    ## 두개의 bounding box가 겹치는 영역의 좌표
    x_left = max(bb1['xmin'], bb2['xmin'])
    x_right = min(bb1['xmax'], bb2['xmax'])
    y_bottom = max(bb1['ymin'], bb2['ymin'])
    y_top = min(bb1['ymax'], bb2['ymax'])

    if x_right < x_left or y_top < y_bottom: 
        return 0

    intersection_area = (x_right - x_left) * (y_top - y_bottom)

    bb1_area = (bb1['xmax'] - bb1['xmin']) * (bb1['ymax'] - bb1['ymin'])
    bb2_area = (bb2['xmax'] - bb2['xmin']) * (bb2['ymax'] - bb2['ymin'])

    iou = intersection_area / (bb1_area + bb2_area - intersection_area)

    assert iou <= 1
    assert iou >= 0
    return iou


def resize(image, x, y, w, h):
    x, y, w, h = int(x), int(y), int(w), int(h)
    # warping with context padding (p = 16 pixels) outperformed the alternatives by a large margin (3-5 mAP points).
    img = image.copy()[max(y - 16, 0):min(image.shape[0], y + h + 16),
                       max(x - 16, 0):min(x + w + 16, image.shape[1])]
    # print(img)
    np.pad(img, (
        (max(0, 16 - y), max(0, image.shape[1] - y - h)), (max(0, 16 - x), max(0, image.shape[1] - x - w)),
        (0, 0)),
           mode='constant', constant_values=(0, 0))
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)  # padding 후 resize

    return img


def detect_pos_or_neg(image, ssresults, gt, threshold):

    t_images = []
    t_labels = []

    pos_num = 0
    neg_num = 0
    for idx, ssbox in enumerate(ssresults):
        if pos_num >= 30 & neg_num >= 30:
            print("break")
            break
        
        # print("pass")
        # print(f"ssbox : {ssbox}")
        x, y, w, h = ssbox
        
        for gt_box in gt:
            # print(gt_box)
            cls, cx, cy, w, h = gt_box.values()
            # print(cls, cx, cy, w, h)
            xmin = cx - w/2
            xmax = cx + w/2
            ymin = cy - h/2
            ymax = cy + h/2
            nbox = {"xmin" : xmin, "xmax" : xmax, "ymin" : ymin, "ymax" : ymax}
        
            img = resize(image, x, y, w, h)

            # limit with 30

            if iou(nbox, ({'xmin': x, 'xmax': x + w, 'ymin': y, 'ymax': y + h})) > threshold:
                if pos_num < 30:
                    t_images.append(img)
                    t_labels.append(int(cls))
                    pos_num += 1
                
            else:
                if neg_num < 30:
                    t_images.append(img)
                    t_labels.append(0)
                    neg_num += 1
    
    print(pos_num, neg_num)
    return t_images, t_labels


def region_proposal(mode):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    train_images = []
    train_labels = []

    if mode == 'finetune':
        threshold = 0.5
    elif mode == 'classify':
        threshold = 0.3
    elif mode == 'test':
        threshold = 0

    for i in os.listdir('./labels'):
        gt = []

        img_path = f'./images/{i[:-4]}.jpg'
        print(f"img_path is {img_path}\n")
        image = cv2.imread(img_path)
        # print(f"image.size = {image.shape}")
        iw, ih, ic = image.shape

        with open(f'./labels/{i}', 'r') as f:
            line = f.readlines()

        for lin in line:
            wordlist = lin.split(" ")
            cls, xcenter, ycenter, w, h = int(wordlist[0]), float(wordlist[1]), float(wordlist[2]), float(wordlist[3]), float(wordlist[4])
            gt.append({"cls" : int(cls), "cx" : int(xcenter * iw), "cy" : int(ycenter * ih), "w" : int(w * iw), "h" : int(h * ih)})

        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        print(len(ssresults))
        # print(ssresults)

        imgs, labels = detect_pos_or_neg(image, ssresults, gt, threshold)

        train_images += imgs
        train_labels += labels
        print(len(train_images))

    return train_images, train_labels


if __name__ == "__main__":
    train_images, train_labels = region_proposal('test')
    print(len(train_images))
    print(train_labels[10])
    plt.imshow(cv2.cvtColor(train_images[10], cv2.COLOR_BGR2RGB))
    plt.show()