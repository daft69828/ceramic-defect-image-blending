from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

img_size = 640
half_img_size = int(img_size / 2)


class Mixer:
    def __init__(self, image_path, mask_path, target_path, top, left, rotation):
        self.image_path, self.mask_path, self.target_path = image_path, mask_path, target_path
        self.top, self.left, self.rotation = top, left, rotation
        self.image = cv2.imread(image_path)
        self.mask = cv2.imread(mask_path)
        self.target = cv2.imread(target_path)

    def expend_image_and_mask(self, height, width, rotation):
        img = self.image
        msk = self.mask
        res_img = self.overlap(img, 300, 1000)
        res_msk = self.overlap(msk, 300, 1000)
        return res_img, res_msk

    def overlap(self, img, height, width):
        background = cv2.imread('./black.png')
        rows, cols = img.shape[:2]
        roi = background[height:height + rows, width:width + cols]

        dst_img = cv2.addWeighted(img, 1.0, roi, 0, 0)

        res_img = background.copy()
        res_img[height:height+rows, width:width+cols] = dst_img

        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        return res_img

    def paste_image(self):
        target = Image.open(self.target_path)
        img = Image.open(r"./temp/temp_result.png")
        height, width = np.array(img).shape[:2]
        pos = [self.top - int(height * 0.5), self.top + math.ceil(height * 0.5), self.left - int(width * 0.5),
               self.left + math.ceil(width * 0.5)]
        # x1, y1 = self.left, self.top
        # x2, y2 = self.left + width, self.top + height
        target.paste(img, (pos[2], pos[0], pos[3], pos[1]))
        # plt.imshow(target)
        # plt.show()
        return pos[2], pos[0], pos[3], pos[1], target

    def resize(self, s, m, type):
        # 不同类型缺陷大中小占比 0边 1角 2白 3浅 4黑 5圈
        ratios = [[0.4, 0.7],       # edge
                  [0.2, 0.5],       # corner
                  [0.33, 0.66],     # white
                  [0.79, 0.93],     # light
                  [0.95, 0.98],     # black
                  [0.66, 0.95]]      # ring
        sizes = [[30, 90, 140, 200],   # edge
                 [45, 50, 160, 300],     # corner
                 [27, 33, 39, 45],      # white
                 [30, 45, 60, 130],     # light
                 [28, 45, 70, 80],      # black
                 [35, 60, 90, 135]]     # ring
        r1 = random.random()
        if r1 <= ratios[type][0]:
            new_size = random.randint(sizes[type][0], sizes[type][1])
        elif ratios[type][0] < r1 <= ratios[type][1]:
            new_size = random.randint(sizes[type][1], sizes[type][2])
        elif r1 > ratios[type][1]:
            new_size = random.randint(sizes[type][2], sizes[type][3])

        r2 = 1
        # if type == 3 and r1 > ratios[type][1]:
        #     r2 = random.uniform(0.45, 0.95)
        # else:
        #     r2 = random.uniform(0.8, 1.0)

        s1 = cv2.resize(s, (new_size, int(new_size * r2)))
        m1 = cv2.resize(m, (new_size, int(new_size * r2)))

        return s1, m1

    def cut_from_target(self, mask):
        target = self.target
        height, width = np.array(mask).shape[:2]

        pos = [self.top-int(height*0.5), self.top+math.ceil(height*0.5), self.left-int(width*0.5), self.left+math.ceil(width*0.5)]
        if 0<pos[0]<640 and 0<pos[1]<640 and 0<pos[2]<640 and 0<pos[3]<640:
            target = target[self.top - int(height * 0.5):self.top + math.ceil(height * 0.5),
                     self.left - int(width * 0.5):self.left + math.ceil(width * 0.5)]
            # target = target[self.top:self.top + height, self.left:self.left + width]
            cv2.imwrite('./temp/cut_target.png', target)
            return target
        else:
            print('超出边界')
            return None




    def update_mask(self, mask, target):
        i = random.randint(1, 10000)
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) # (51, 51)
        ret, thresh = cv2.threshold(blurred, 69, 255, cv2.THRESH_BINARY)

        # print(mask.shape)
        # # cv2.imshow('2', mask[int(mask.shape[0] * 0.35): int(mask.shape[0] * 0.65),
        # #                                   int(mask.shape[1] * 0.35): int(mask.shape[1] * 0.65)])
        # cv2.imshow('2', mask[100: 200, 90: 180])
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        edge = thresh - cv2.erode(thresh, np.ones((3, 3), np.uint8))
        non_zero_count = cv2.countNonZero(edge[int(mask.shape[0] * 0.35): int(mask.shape[0] * 0.65),
                                          int(mask.shape[1] * 0.35): int(mask.shape[1] * 0.65)])
        # if 15 <= non_zero_count <= 50:
        if True:
            res_mask = cv2.bitwise_and(mask, thresh)
            four = np.hstack((gray, thresh, mask, res_mask))
            cv2.imwrite('./results/four/' + str(i) + '.png', four)
            # print('non_zero_count:', non_zero_count)
            return res_mask
        else:
            return None
