import cv2
from utils import yolo_detect_api
from lxml.etree import Element, SubElement, tostring
import hashlib


def create_xml(list_xml, list_images, xml_path):
    """
    创建xml文件，依次写入xml文件必备关键字
    :param list_xml:   txt文件中的box
    :param list_images:   图片信息，xml中需要写入WHC
    :return:
    """
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'Images'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(list_images[3])
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(list_images[1])
    node_height = SubElement(node_size, 'height')
    node_height.text = str(list_images[0])
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(list_images[2])

    # 按目标循环写入标签信息
    if len(list_xml) >= 1:
        for list_ in list_xml:
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            ''' 
            # ---标注特定类别，不符合则跳过---
            if str(list_[4]) == "person":
                node_name.text = str(list_[4])
            else:
                continue
            '''
            node_name.text = str(list_[4])
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(list_[0])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(list_[1])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(list_[2])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(list_[3])

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行

    file_name = list_images[3].split(".")[0]
    filename = xml_path + "/{}.xml".format(file_name)

    f = open(filename, "wb")
    f.write(xml)
    f.close()


# 获取文件MD5值
def get_md5(file_name, path):
    with open(os.path.join(path, file_name), 'rb') as f:
        md5obj = hashlib.md5()
        md5obj.update(f.read())
        hash = md5obj.hexdigest()
        return hash


# 查找目标目录下的所有文件,并使用MD5值及后缀名重命名当前文件
def rename_md5(path):
    winerror = []
    label_name = opt.rename + "_"
    for root, dirlist, filelist in os.walk(path):
        for file in filelist:
            newname = label_name + '{0}.{1}'.format(get_md5(file, path), file.split('.')[-1])
            # print(newname)
            try:
                if os.path.join(path, newname) == os.path.join(path, file):
                    pass
                else:
                    print('Now Renaming:', file, 'To', newname)
                    os.rename(os.path.join(path, file), os.path.join(path, newname))
            except WindowsError:
                nickname = '{0}.{1}'.format(str(len(winerror)), file.split('.')[-1])
                print('WindowsError for:', file, 'Renaming to:', nickname)
                winerror.append(file)
                os.rename(os.path.join(path, file), os.path.join(path, nickname))


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="", help='initial weights path')
    parser.add_argument('--img-path', type=str, default='../data/auto_label')
    parser.add_argument('--xml-path', type=str, default='../data/auto_label/xml', help='xml save_path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--rename', type=str, default='', help='rename into label_hd5')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save-pred', action='store_true', help='save picture of pred')
    opt = parser.parse_args()

    if not os.path.exists(opt.xml_path):
        os.makedirs(opt.xml_path)
    if opt.rename != '':
        rename_md5(opt.img_path)
    yolo = yolo_detect_api.yolo_inference()

    for name in os.listdir(opt.img_path):
        print(name)
        pred_path = os.path.join(opt.xml_path, name.split(".")[0]) + '_pred.jpg'
        image = cv2.imread(os.path.join(opt.img_path, name))
        try:
            list_image = (image.shape[0], image.shape[1], image.shape[2], name)  # 图片的宽高等信息
        except:
            continue
        img0, xyxy_list, img_crop = yolo.detect(opt.weights, image, opt.img_size, opt.conf_thres, opt.iou_thres)
        if opt.save_pred:
            cv2.imwrite(pred_path, img0)  # 保存预测图片至xml本地，直观查看标注框
        create_xml(xyxy_list, list_image, opt.xml_path)  # 生成标注的xml文件
