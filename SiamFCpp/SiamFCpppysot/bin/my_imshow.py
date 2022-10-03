import cv2
import numpy as np

def show_multi_imgs(scale, imglist, order=None, border=10, border_color=(255, 255, 0)):
    """
    :param scale: float 原图缩放的尺度
    :param imglist: list 待显示的图像序列
    :param order: list or tuple 显示顺序 行×列
    :param border: int 图像间隔距离
    :param border_color: tuple 间隔区域颜色
    :return: 返回拼接好的numpy数组
    """
    if order is None:
        order = [1, len(imglist)]
    allimgs = imglist.copy()
    ws , hs = [], []
    for i, img in enumerate(allimgs):
        if np.ndim(img) == 2:
            allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        ws.append(allimgs[i].shape[1])
        hs.append(allimgs[i].shape[0])
    w = max(ws)
    h = max(hs)
    # 将待显示图片拼接起来
    sub = int(order[0] * order[1] - len(imglist))
    # 判断输入的显示格式与待显示图像数量的大小关系
    if sub > 0:
        for s in range(sub):
            allimgs.append(np.zeros_like(allimgs[0]))
    elif sub < 0:
        allimgs = allimgs[:sub]
    imgblank = np.zeros(((h+border) * order[0], (w+border) * order[1], 3)) + border_color
    imgblank = imgblank.astype(np.uint8)
    for i in range(order[0]):
        for j in range(order[1]):
            imgblank[(i * h + i*border):((i + 1) * h+i*border), (j * w + j*border):((j + 1) * w + j*border), :] = allimgs[i * order[1] + j]
    return imgblank

def imgsLen(scene):
    data = np.loadtxt("F:\Datasets\RobotVision\RealValImg" + '/' + scene + '\\' + 'groundtruth.txt', delimiter=',')
    return len(data)

def markImg(trakerName, scene, num):
    groundtruthDir = "F:\Datasets\RobotVision\RealValImg"
    imgDir = "F:\Datasets\RobotVision\RealValImg"
    boundingBoxDir = "D:\Projects\PycharmProject\ObjectTracking\SiamFCpp\SiamFCpppysot\\results\RealValImg"
    scene = scene
    num = num
    bboxList = []

    trackerName = trakerName
    textDir = boundingBoxDir + '\\' + trackerName + '\\baseline\\' + scene + '\\' \
              + scene.split('\\')[0] + scene.split('\\')[1] + '_001.txt'
    f = open(textDir, 'r')
    for row in f.readlines():
        item = row.split(',')
        if len(item) == 1:
            bboxList.append([0, 0, 0, 0])
        else:
            bboxList.append(list(map(float, item)))

    data = np.loadtxt(groundtruthDir + '/' + scene + '\\' + 'groundtruth.txt', delimiter=',')

    gt = data[num]
    bbox = bboxList[num]
    img = cv2.imread(imgDir + '\\' + scene + '\\' + str(num) + '.jpg')
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                        (0, 255, 255), 2)
    img = cv2.rectangle(img, (int(gt[0]), int(gt[1])), (int(gt[0] + gt[2]), int(gt[1] + gt[3])), (0, 255, 0), 2)
    img = cv2.putText(img, trakerName, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    return img



if __name__ == '__main__':
    scene = r"workpiece7\scene04"
    num = 1
    trakers = ["siamcar_alex_real", "siamcar_alex_virtual", "siamcar_alex_mix", "siamcar_conv_real",
               "siamcar_conv_virtual", "siamcar_conv_mix"]

    for i in range(1, imgsLen(scene)):
        imgs = []

        for traker in trakers:
            imgs.append(markImg(traker, scene, i))

        multi = show_multi_imgs(0.5, imgs, (2, 3), border_color=(255, 255, 255))
        cv2.imshow(" ", multi)
        cv2.waitKey()


