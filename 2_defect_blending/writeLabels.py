from xml.dom.minidom import Document
from xml.dom import minidom
import os
import cv2


class LabelWriter():
    def __init__(self, xmin, ymin, xmax, ymax, savedir, savename, mskdir):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.savedir = savedir
        self.savename = savename
        self.mskdir = mskdir

    def findBbox(self):
        src_path = self.mskdir + self.savename[:-4] + '.png'

        img_src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        ret, img_bi = cv2.threshold(img_src, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(img_bi, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)
        img_result = img_src.copy()
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        area = 0
        [x1, y1, w1, h1] = [0, 0, 0, 0]
        for bbox in bounding_boxes:
            [x, y, w, h] = bbox
            if w * h >= area:
                [x1, y1, w1, h1] = [x, y, w, h]
                area = w * h
        # cv2.rectangle(img_result, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), 1)
        return [x1, y1, x1 + w1, y1 + h1]

    def writeLabel(self, defect_type, bbox):
        doc = Document()

        label = doc.createElement("annotation")
        doc.appendChild(label)

        filename = doc.createElement("filename")
        label.appendChild(filename)
        text_filename = doc.createTextNode(self.savename)
        filename.appendChild(text_filename)

        size = doc.createElement("size")
        label.appendChild(size)
        width = doc.createElement("width")
        size.appendChild(width)
        text_width = doc.createTextNode("640")
        width.appendChild(text_width)
        height = doc.createElement("height")
        size.appendChild(height)
        text_height = doc.createTextNode("640")
        height.appendChild(text_height)
        depth = doc.createElement("depth")
        size.appendChild(depth)
        text_depth = doc.createTextNode("3")
        depth.appendChild(text_depth)

        object = doc.createElement("object")
        label.appendChild(object)
        name = doc.createElement("name")
        object.appendChild(name)
        text_name = doc.createTextNode(defect_type)
        name.appendChild(text_name)
        bndbox = doc.createElement("bndbox")
        object.appendChild(bndbox)
        xmin = doc.createElement("xmin")
        bndbox.appendChild(xmin)
        text_xmin = doc.createTextNode(str(bbox[0] + self.xmin))
        xmin.appendChild(text_xmin)
        ymin = doc.createElement("ymin")
        bndbox.appendChild(ymin)
        text_ymin = doc.createTextNode(str(bbox[1] + self.ymin))
        ymin.appendChild(text_ymin)
        xmax = doc.createElement("xmax")
        bndbox.appendChild(xmax)
        text_xmax = doc.createTextNode(str(bbox[2] + self.xmin))
        xmax.appendChild(text_xmax)
        ymax = doc.createElement("ymax")
        bndbox.appendChild(ymax)
        text_ymax = doc.createTextNode(str(bbox[3] + self.ymin))
        ymax.appendChild(text_ymax)

        filename = self.savedir + self.savename
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        f = open(filename, "w", encoding='utf-8')
        f.write(doc.toprettyxml(indent="  "))
        f.close()
        return [bbox[0] + self.xmin, bbox[1] + self.ymin,
                bbox[2] + self.xmin, bbox[3] + self.ymin]

    def readLabel(self, xmlPath):
        dom = minidom.parse(xmlPath)
        root = dom.documentElement

        elem_x1 = root.getElementsByTagName('xmin')
        elem_y1 = root.getElementsByTagName('ymin')
        elem_x2 = root.getElementsByTagName('xmax')
        elem_y2 = root.getElementsByTagName('ymax')
        x1 = int(elem_x1[0].firstChild.data)
        y1 = int(elem_y1[0].firstChild.data)
        x2 = int(elem_x2[0].firstChild.data)
        y2 = int(elem_y2[0].firstChild.data)

        return [x1, y1, x2, y2]







