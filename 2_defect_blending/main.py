import os.path
import cv2
from poisson_image_editing import poisson_edit
from mix import Mixer
from writeLabels import LabelWriter
import getopt
import sys
from os import path
import numpy as np
from PIL import ImageDraw
import random
import re
import math
import argparse

img_size = 640
half_img_size = int(img_size / 2)
defect_dic = {0: "边异常", 1: "角异常", 2: "白色点瑕疵", 3: "浅色块瑕疵", 4: "深色点块瑕疵", 5: "光圈瑕疵", 6: "calculate test"}

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--category', '-c', type=int, default="0", required=True, help="缺陷类型（0-6）分别表示：边异常、角异常、白色点瑕疵、浅色块瑕疵、深色点块瑕疵、光圈瑕疵")
    parser.add_argument('--quantity', '-q', type=int, default=10, help='得到的结果数量（若已有数量超过该值，直接结束）')
    args = parser.parse_args()
    defect_type = args.category
    total_samples = args.quantity

    # 融合时所需的sources和对应masks的保存位置
    src_dirs = ['./sources_n_masks/edge/',
                './sources_n_masks/corner/',
                './sources_n_masks/whitepoint/',
                './sources_n_masks/lightcolorpoint/',
                './sources_n_masks/blackpoint/',
                './sources_n_masks/ring/',
                './sources_n_masks/calculate_test/']
    sources_dir = src_dirs[defect_type] + 'img/'
    mask_dir = src_dirs[defect_type] + 'msk/'

    # 融合时所需的target的保存位置
    if defect_type == 0:
        targets_dir = './targets/edge/'
    elif defect_type == 1:
        targets_dir = './targets/corner/'
    elif defect_type == 6:
        targets_dir = './targets/calculate_test/'
    else:
        targets_dir = './targets/other_categories/'

    # 融合后结果的保存位置
    dst_dirs = ['./results/edge/',
                './results/corner/',
                './results/whitepoint/',
                './results/lightcolorpoint/',
                './results/blackpoint/',
                './results/ring/',
                './results/calculate_test']
    img_save_dir = dst_dirs[defect_type] + 'images/'
    label_save_dir = dst_dirs[defect_type] + 'labels/'
    msk_save_dir = dst_dirs[defect_type] + 'mask4labels/'
    img_with_bbox_save_dir = dst_dirs[defect_type] + 'imgWithB/'
    temp_dir = './temp/'
    # target_save_dir = './targets_for_edge_640/'

    if not os.path.exists(img_save_dir): os.makedirs(img_save_dir)
    if not os.path.exists(msk_save_dir): os.makedirs(msk_save_dir)
    if not os.path.exists(label_save_dir): os.makedirs(label_save_dir)
    if not os.path.exists(img_with_bbox_save_dir): os.makedirs(img_with_bbox_save_dir)
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)

    sourcesList = sorted_aphanumeric(os.listdir(sources_dir))
    # sourcesList.reverse()
    total_sources = len(sourcesList)
    target_list = os.listdir(targets_dir)
    while True:
        for fileName in sourcesList:
            if len(os.listdir(img_save_dir)) >= total_samples:
                print('Total imgs in save dirs:', len(os.listdir(img_save_dir)), '/', total_samples)
                sys.exit()

            i = random.randrange(0, len(target_list))
            source_path = sources_dir + fileName
            mask_path = mask_dir + fileName
            target_path = targets_dir + target_list[i]

            source = cv2.imread(source_path)
            mask = cv2.imread(mask_path)
            target = cv2.imread(target_path)

            img_save_name = os.path.basename(fileName)[:-4] + '.png'
            label_save_name = os.path.basename(fileName)[:-4] + '.xml'
            img_save_path = img_save_dir + img_save_name
            msk_save_path = msk_save_dir + img_save_name
            label_save_path = label_save_dir + label_save_name
            img_with_bbox_save_path = img_with_bbox_save_dir + img_save_name
            # print('img_save_path:', img_save_path)
            # print('msk_save_path:', msk_save_path)
            # print('label_save_path:', label_save_path)
            # os.system("pause")

            # 0. find a place
            # (1) for edge
            if defect_type == 0:
                temp = target.copy()
                gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (11, 11), 0)
                edge = cv2.Canny(blurred, 30, 90)
                lines = cv2.HoughLinesP(edge, 1, np.pi / 180, 5, minLineLength=400, maxLineGap=50)
                if lines is None: continue
                max_distance = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    distance = math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
                    if distance > max_distance:
                        max_distance = distance
                        longest = line[0]
                x1, y1, x2, y2 = longest
                mid_y, mid_x = int((x1 + x2) / 2), int((y1 + y2) / 2)
                print("position:", mid_x, mid_y)
                rand_offset = random.randint(20, 40)
                if math.fabs(x2 - x1) > math.fabs(y2 - y1):
                    # print('--', gray[int(mid_x/2), mid_y], gray[int((mid_x + 640)/2), mid_y])
                    if gray[int(mid_x / 2), mid_y] < gray[int((mid_x + 640) / 2), mid_y]:  # 上背景下瓷砖
                        pos_y, pos_x = int((x1 + x2) / 2) + rand_offset, int((y1 + y2) / 2)
                    else:  # 上瓷砖下背景
                        pos_y, pos_x = int((x1 + x2) / 2) - rand_offset, int((y1 + y2) / 2)
                else:
                    # print('|', gray[mid_x, int(mid_y / 2)], gray[mid_x, int((mid_y + 640) / 2)])
                    if gray[mid_x, int(mid_y / 2)] < gray[mid_x, int((mid_y + 640) / 2)]:  # 左背景右瓷砖
                        pos_y, pos_x = int((x1 + x2) / 2), int((y1 + y2) / 2) + rand_offset
                    else:
                        pos_y, pos_x = int((x1 + x2) / 2), int((y1 + y2) / 2) - rand_offset

                # pos_y, pos_x = int((x1 + x2) / 2), int((y1 + y2) / 2)
                # print('mid line pos:', pos_x, pos_y)
                # cv2.line(target, (x1, y1), (x2, y2), (100, 200, 100), 2)
                # cv2.circle(target, (pos_y, pos_x), 20, (0, 255, 255), 3)
                # gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                # edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
                # three = np.hstack((gray_rgb, edge_rgb, target))
                # cv2.imshow('gray/edge/img', three)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

            # (2) for corner
            elif defect_type == 1:
                gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (35, 35), 0)
                ret, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
                corners = cv2.goodFeaturesToTrack(gray, 1, 0.03, 100, np.ones((thresh.shape[0], thresh.shape[1])))
                pos_y, pos_x = int(corners[0][0][0]), int(corners[0][0][1])
                rand_offset_y = random.randint(-5, 5)
                rand_offset_x = random.randint(-5, 5)
                pos_y, pos_x = pos_y + rand_offset_y, pos_x + rand_offset_x
                print("position:", pos_x, pos_y)
                # cv2.circle(target, (int(corners[0][0][0]), int(corners[0][0][1])), 20, (0, 0, 255), 3)
                # two = np.hstack((cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), img1))
                # cv2.imshow('thresh/img', two)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
            # (3) for test
            elif defect_type == 6:
                pos_y, pos_x = half_img_size, half_img_size
                print('pos:', pos_x, pos_y)
            # (4) for others
            else:
                pos_y, pos_x = random.randint(100, 540), random.randint(100, 540)
                print("position:", pos_x, pos_y)

            # 1. resize source, target and mask
            mixer = Mixer(source_path, mask_path, target_path, pos_x, pos_y, 0)
            if defect_type != 6:
                source, mask = mixer.resize(source, mask, defect_type)

            # 2. cut target
            cut_target = mixer.cut_from_target(mask)
            if cut_target is None:
                print('cut_target is None, continue')
                continue

            # 3. update mask for type 0 and 1
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            # cv2.imshow('1', mask)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            if defect_type == 0 or defect_type == 1:
                mask = mixer.update_mask(mask, cut_target)

            # 4. save mask
            if mask is not None:
                while os.path.exists(msk_save_path):
                    oldName = fileName
                    prefix, postfix = fileName.split('.')
                    figure = int(prefix[4:])
                    figure += total_sources
                    fileName = 'Img_' + str(figure) + '.' + postfix
                    msk_save_path = msk_save_dir + fileName
                    print('[' + oldName + '] exists, try saving as [' + fileName + ']!')
                cv2.imwrite(msk_save_path, mask)
                print('[' + fileName + '] ' + 'saved!')
                # cv2.imwrite(target_save_dir + os.path.basename(args["target"]), target)
            else:
                continue

            # 5. blend
            offset = (0, 0)
            poisson_blend_result = poisson_edit(source, cut_target, mask, offset)
            # poisson_blend_result = cv2.addWeighted(source, 1, cut_target, 0, 0)
            cv2.imwrite(path.join(temp_dir, 'temp_result.png'), poisson_blend_result)

            # 6. paste and save
            x1, y1, x2, y2, paste_result = mixer.paste_image()
            # draw = ImageDraw.Draw(paste_result)
            # draw.rectangle([x1, y1, x2, y2], outline="yellow")
            paste_result.save(img_save_dir + fileName)

            # 7. write labels
            # print(x1, y1, x2, y2)
            label_save_name = fileName[:-4] + '.xml'
            labelWriter = LabelWriter(x1, y1, x2, y2, label_save_dir, label_save_name, msk_save_dir)
            bbox_in = labelWriter.findBbox()
            bbox_out = labelWriter.writeLabel(defect_dic[defect_type], bbox_in)

            # 8. read labels and draw
            bbox_label = labelWriter.readLabel(label_save_dir + label_save_name)
            draw = ImageDraw.Draw(paste_result)
            draw.rectangle([bbox_label[0], bbox_label[1], bbox_label[2], bbox_label[3]], outline="blue")
            paste_result.save(img_with_bbox_save_dir + fileName)

            paste_result = cv2.cvtColor(np.asarray(paste_result), cv2.COLOR_RGB2BGR)


    print('Done.\n')

    # print('Targets picked:', len(os.listdir(target_save_dir)))
    # cv2.imshow('paste_result', paste_result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
