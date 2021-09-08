from numpy import *
import os
import time
import onnxruntime as ort
from utils.onnx import *
import argparse


anchors = [[10, 13], [16, 30], [33, 23],
           [30, 61], [62, 45], [59, 119],
           [116, 90], [156, 198], [373, 326]]

# 模型输入尺寸
model_input_h = 640
model_input_w = 640
# 类别数量
classes_num = 3
# 类别标签
class_label = 'dmcode', 'qrcode', 'barcode'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default=r'', help='model.onnx path(s)')
    parser.add_argument('--img', type=str, default=r'', help='img directory')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--conf-thres', '--conf', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', '--iou', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--project', default='runs/onnx', help='save results to project/name')
    parser.add_argument('--val', action='store_true', help='val_pre between pth and onnx')
    parser.add_argument('--tensor', type=str, default='', help='tensor_output path')
    opt = parser.parse_args()

    if not os.path.exists(os.path.join(opt.project, opt.name)):
        os.makedirs(os.path.join(opt.project, opt.name))
    if opt.tensor != '' and not os.path.exists(opt.tensor):
        os.makedirs(opt.tensor)
    save_dir = os.path.join(opt.project, opt.name)
    index =0
    inferece_time =[]

    # 读取图片并转换到RGB图像格式,
    # 注意opencv图像维度顺序为：
    # shape[0]=h, shape[1]=w, shape[2]=c
    for name in os.listdir(opt.img):
        index += 1
        image_path = os.path.join(opt.img, name)
        save_path = os.path.join(save_dir, name.split(".")[0]) + '.jpg'
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            print('load image {} failed !!!'.format(image_path))
            exit(-1)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # 图像尺寸是否与模型输入尺寸一致，如果不一致则进行填充、缩放
        if bgr_image.shape != (model_input_h, model_input_w, 3):
            rgb_image = pad_to_square(rgb_image)
            rgb_image = cv2.resize(rgb_image, (model_input_w, model_input_h))

        # 图像数据类型转换为浮点类型，保证模型的输入为浮点类型
        rgb_image = rgb_image.astype(np.float32)

        # 归一化
        rgb_image = rgb_image / 255.0

        # 维度变换：HWC => CHW
        rgb_image = rgb_image.transpose(2, 0, 1)

        # 增加一个数据维度：CHW => NCHW
        input_data = np.expand_dims(rgb_image, axis=0)

        print('img_name = {}'.format(name))
        print('input_data shape = {}'.format(input_data.shape))
        # print('input_data type = {}'.format(input_data.dtype))

        # 启动onnx runtime会话
        ort_session = ort.InferenceSession(opt.onnx)

        start = time.time()

        # 模型运行
        model_outputs = ort_session.run(None, {'images': input_data})

        end = time.time()
        time_ms = (end - start) * 1000
        print('inference time: {} ms'.format(time_ms))

        # 保存模型输出结果, 打印模型输出维度
        if opt.tensor != '':
            for index, output in enumerate(model_outputs):
                save_tensor(model_outputs[index], os.path.join(opt.tensor, 'onnx', 'output{}'.format(index)))
                print('yolo output {} shape: {}'.format(index, output.shape))

        # 模型输出解码
        decode_outputs = decode(model_outputs=model_outputs,
                                model_input_h=model_input_h,
                                model_input_w=model_input_w,
                                anchors=anchors,
                                cls_num=classes_num)

        print(decode_outputs.shape)
        # 非极大值抑制（NMS）
        output = non_max_suppression(decode_outputs,
                                     classes_num=classes_num,
                                     obj_conf_thresh=opt.conf_thres,
                                     nms_iou_thresh=opt.iou_thres)
        '''
        # 将坐标还原至原图尺寸
        boxs = output[0][..., :4]
        h, w, c = bgr_image.shape
        boxs[:, :] = scale_coords((model_input_h, model_input_w), boxs[:, :], (h, w)).round()
        '''

        # 绘制结果
        draw_result(class_label,
                    detections=output[0],
                    image_src=bgr_image,
                    input_size=(model_input_h, model_input_w),
                    line_thickness=None,
                    text_bg_alpha=0.0)

        # 保存结果
        # cv2.imwrite('C:/Users/lukaijie/Desktop/test_onnx/result/onnx_{}'.format(image_name), bgr_image)
        cv2.imwrite(save_path, bgr_image)

        # 保存日记
        log = save_dir + '_inference.txt'
        with open(log, 'a') as f:
            f.write('#######pic{}#######'.format(index) + '\n')
            f.write('img_name = {}'.format(name) + '\n')
            f.write('inference time: {} ms'.format(time_ms) + '\n')
        f.close()
        inferece_time.append(time_ms)

        # 记录平均耗时
        if index == len(os.listdir(opt.img)):
            sum = 0
            for i in inferece_time:
                sum = sum + i
            with open(log, 'a') as f:
                f.write('平均耗时：{}ms'.format(sum/index) + '\n')
            f.close()
