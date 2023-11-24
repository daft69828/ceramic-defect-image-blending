import json, os
import os.path
from PIL import Image
from PIL import ImageFile
import re
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

# 0边 1角 2白 3浅 4黑 5圈
extract_type = 5
src_dir = './add/test_and_val/images/test_and_val/'
dst_dir = './add/test_and_val/images/cropped/ring/'
label_dir = './add/test_and_val/labels/test_and_val/'
dst_label_dir = './add/test_and_val/labels/cropped/ring/'
if not os.path.exists(dst_dir): os.makedirs(dst_dir)
if not os.path.exists(dst_label_dir): os.makedirs(dst_label_dir)

count = 0
count_not_extract = 0
flag = 0 # 该图片是否含有所选类型的缺陷
img_list = sorted_aphanumeric(os.listdir(src_dir))

for name in img_list:
    src_path = src_dir + name
    # dst_path = dst_dir + 'Img_' + str(count) + '.jpg'
    dst_path = dst_dir + 'x_' + name
    dst_label_path = dst_label_dir + 'x_' + name[:-4] + '.txt'
    #print(dst_path, dst_label_path)
    # os.system("pause")
    label_path = label_dir + name[:-4] + '.txt'
    img = Image.open(src_path)
    print(label_path)
    with open(label_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for i in range(len(content)):
            info = list(map(float, content[i].split()))
            def_type = int(info[0])
            x1 = info[1] * 640 - info[3] * 640 / 2
            y1 = info[2] * 640 - info[4] * 640 / 2
            x2 = info[1] * 640 + info[3] * 640 / 2
            y2 = info[2] * 640 + info[4] * 640 / 2
            if def_type == extract_type:
                flag = 1
                # print(def_type, x1, x2, y1, y2)
                # x_mid = int((x1 + x2) / 2)
                # y_mid = int((y1 + y2) / 2)
                # x1 = x_mid - 64
                # y1 = y_mid - 64
                # x2 = x_mid + 64
                # y2 = y_mid + 64
                # if x1 < 0 or x2 > 640 or y1 < 0 or y2 > 640:
                #     count_not_extract += 1
                #     continue
                # defect_area = img.crop((x1, y1, x2, y2))
                # defect_area.save(dst_path, quality=95, subsampling=0)
                count += 1
        if flag:
            flag = 0
            shutil.copyfile(src_path, dst_path)
            shutil.copyfile(label_path, dst_label_path)
print('extracted:', count)
print('not extracted:', count_not_extract)
