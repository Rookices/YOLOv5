import xml.etree.ElementTree as ET
import os
import shutil


def check_size(b, newfile, single=True, move=True):
    path = b
    # 图片移动新路径
    for name in os.listdir(path):
        if name.split('.')[-1] == 'xml':
            log = 0
            xml_path = path + '/' + name
            tree = ET.parse(xml_path)
            root = tree.getroot()
            """ 图片size有误           
            for object_1 in root.iter("size"):
                if (object_1.find("width").text != 0) and (object_1.find("height").text != 0):
                    continue
                else:
                    log += 1
            """
            if single:
                # 常规单类
                for object_1 in root.iter("object"):
                    if object_1.find("name").text == 'dmcode':
                        continue
                    else:
                        log += 1
            else:
                # 常规多类
                label = {}
                dmcode, qrcode, barcode = {0: 'dmcode'}, {1: 'qrcode'}, {2: 'barcode'}
                for object_1 in root.iter("object"):
                    if object_1.find("name").text == 'dmcode':
                        label.update(dmcode)
                        continue
                    if object_1.find("name").text == 'qrcode':
                        label.update(qrcode)
                        continue
                    else:
                        label.update(barcode)
                        continue
                if len(label) < 2:
                    log += 1
            if log > 0:
                if move:
                    # ----移动图片并删除对应xml----
                    jpg_path = xml_path.split('.')[0] + '.jpg'
                    os.remove(xml_path.split('.')[0] + '.xml')
                    shutil.move(jpg_path, os.path.join(newfile, name.split('.')[0] + '.jpg'))
                else:
                    # ----记录标签有误样本----
                    worse_txt = xml_dir + 'worse_test.txt'
                    with open(worse_txt, 'a') as f:
                        f.write(xml_path + '\n')
                    f.close()


xml_dir = r''
# 图片移动新路径
newfile = r''
check_size(xml_dir, newfile, single=True, move=False)
