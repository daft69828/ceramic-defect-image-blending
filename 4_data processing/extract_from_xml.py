import json, os
import os.path
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

label_path = r"train_imgs"

json_path = 'train_annos.json'

defect_list = ["background", "edge_defect", "corner_defect",
               "white_point", "light_color_point", "dark_color_point", "light_ring"]

count = 0
count_not_extract = 0
defect_num = [0, 0, 0, 0, 0, 0, 0]

data_list = json.load(open(json_path, 'r'))
print(len(data_list))
for data_dict in data_list:
    save_path = "new_extract"
    img_height = data_dict['image_height']
    img_width = data_dict['image_width']
    defect_category = data_dict['category']
    img_name = data_dict['name']

    defect_num[defect_category] += 1

    if defect_category == 5:
        x1 = data_dict['bbox'][0]
        y1 = data_dict['bbox'][1]
        x2 = data_dict['bbox'][2]
        y2 = data_dict['bbox'][3]

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x_mid = int((x1 + x2) / 2)
        y_mid = int((y1 + y2) / 2)
        x1 = x_mid - 64
        y1 = y_mid - 64
        x2 = x_mid + 64
        y2 = y_mid + 64
        if x1 < 0 or x2 > img_width or y1 < 0 or y2 > img_height:
            count_not_extract += 1
            continue

        # x1 = x1 - 10 if x1 > 10 else 0
        # y1 = y1 - 10 if y1 > 10 else 0
        # x2 = x2 + 10 if x2 < img_width else img_width
        # y2 = y2 + 10 if y2 < img_height else img_height

        ## 保存图片
        img = Image.open(os.path.join(label_path, img_name))
        # print(os.path.join(label_path, img_name))
        defect_area = img.crop((x1, y1, x2, y2))

        samples_of_image = 0
        save_path = os.path.join(save_path, defect_list[defect_category])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        while os.path.isfile(
                os.path.join(save_path, os.path.splitext(img_name)[0] + '_' + str(samples_of_image) + '.jpg')):
            samples_of_image = int(samples_of_image)
            samples_of_image += 1

        img_save_path = os.path.join(save_path,
                                     os.path.splitext(img_name)[0] + '_' + str(samples_of_image) + '.jpg')
        print(img_save_path)
        print('(' + str(count) + '/' + str(len(data_list)) + ')')
        defect_area.save(img_save_path, quality=95, subsampling=0)
    else:
        print('category:' + defect_list[defect_category] + '(skip)')
        print('(' + str(count) + '/' + str(len(data_list)) + ')')
    count += 1
print(defect_num)