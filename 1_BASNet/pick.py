import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

# 需要筛选的img和msk位置
dir_ori = '../stylegan3/out_new/ring_pick_999/'
dir_pd_mask = '../stylegan3/out_new/ring_pick_999_mask/'
# 结果保存位置
dir_res_img = '../stylegan3/out_new/ring_pick_999_pick_4/'
dir_res_msk = '../stylegan3/out_new/ring_pick_999_pick_mask_4/'

win_size = 128 * 128
sum = 0

if not os.path.exists(dir_res_img): os.makedirs(dir_res_img)
if not os.path.exists(dir_res_msk): os.makedirs(dir_res_msk)

count_img = len(os.listdir(dir_res_img))

root_path = os.listdir(dir_ori)
root_path.sort(key = lambda x:int(x[4:-4]))

for file in root_path:
    path_ori = os.path.join(dir_ori, file)
    path_pd_mask = os.path.join(dir_pd_mask, file)

    img_ori = cv2.imread(path_ori)
    img_pd_mask = cv2.imread(path_pd_mask)

    gray = cv2.cvtColor(img_pd_mask, cv2.COLOR_RGB2GRAY)

    non_zero_counts = cv2.countNonZero(gray)
    zero_counts = win_size - non_zero_counts
    zero_frac = float(zero_counts) / win_size
    # print(file, zero_frac)
    # if zero_frac >= 0.79 or zero_frac <= 0.05:
    #     print(str(zero_frac) + ' next one~')
    #     continue

    while True:
        two = np.hstack((img_ori, img_pd_mask))
        windowname = file + ' 1=save, 3=abort, q=quit ' + 'We got ' + str(count_img) + ' images already!'
        cv2.namedWindow(windowname, 0);
        cv2.resizeWindow(windowname, 640, 640);
        cv2.moveWindow(windowname, 640, 220)
        cv2.imshow(windowname, two)

        # cv2.imwrite(dir_res_img + file, img_ori, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # print(file + ' img saved!')
        # cv2.imwrite(dir_res_msk + file, img_pd_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # print(file + ' msk saved!')
        # cv2.destroyAllWindows()
        # count_img += 1
        # break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("1"):
            cv2.imwrite(dir_res_img + file, img_ori, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(dir_res_msk + file, img_pd_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(file + ' saved!')
            cv2.destroyAllWindows()
            count_img += 1
            break
        elif key == ord("3"):
            print(file + ' aborted!')
            cv2.destroyAllWindows()
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()
