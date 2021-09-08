import cv2
import numpy as np
import torch


def save_tensor(tensor, file_name):
    np.savetxt(file_name, tensor.reshape((-1, 1)))


def pad_to_square(image, pad_value=(114, 114, 114)):
    h, w, c = image.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = (0, 0, pad1, pad2) if h >= w else (pad1, pad2, 0, 0)
    image_pad = cv2.copyMakeBorder(image, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=pad_value)
    return image_pad


def decode(model_outputs, model_input_h, model_input_w, anchors, cls_num):
    total_layer_output = []

    # 调整anchors的维度，方便与后面维度为
    # (1, 3, x, x, 2)的out[..., 2:4]相乘
    # a shape => (3, 3, 2)
    # anchor_grid shape => (3, 1, 3, 1, 1, 2)
    # anchor_grid元素的shape =>（1, 3, 1, 1, 2）
    a = torch.tensor(anchors).float().view(3, -1, 2)
    anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)

    for index, out in enumerate(model_outputs):
        out = torch.from_numpy(out)
        feature_h = out.shape[2]  # 80 or 40 or 20
        feature_w = out.shape[3]  # 80 or 40 or 20

        # 特征图相对于原始图片的缩放因子
        stride_w = int(model_input_w / feature_w)
        stride_h = int(model_input_h / feature_h)

        # 生成80x80 or 40x40 or 20x20的网格
        grid_x, grid_y = np.meshgrid(np.arange(feature_w), np.arange(feature_h))
        grid_x, grid_y = torch.from_numpy(np.array(grid_x)).float(), torch.from_numpy(np.array(grid_y)).float()

        # 目标框解码，得到bx、by、bw、bh
        pred_boxes = torch.FloatTensor(out[..., :4].shape)
        # bx = 2 * sigmoid(tx) - 0.5 + Cx
        pred_boxes[..., 0] = (2.0 * torch.sigmoid(out[..., 0]) - 0.5 + grid_x) * stride_w
        # by = 2 * sigmoid(ty) - 0.5 + Cy
        pred_boxes[..., 1] = (2.0 * torch.sigmoid(out[..., 1]) - 0.5 + grid_y) * stride_h
        # bw = Pw * (2 * sigmoid(tw))^2 or bh = Ph * (2 * sigmoid(th))^2
        pred_boxes[..., 2:4] = (torch.sigmoid(out[..., 2:4]) * 2) ** 2 * anchor_grid[index]

        # 目标置信度解码
        conf = torch.sigmoid(out[..., 4])

        # 类别置信度解码
        pred_cls = torch.sigmoid(out[..., 5:])

        # 拼接该层的结果，output shape => (1, obj_nums, 85)
        output = torch.cat((pred_boxes.view(1, -1, 4),
                            conf.view(1, -1, 1),
                            pred_cls.view(1, -1, cls_num)),
                           -1)
        total_layer_output.append(output)
    # 拼接三个层的结果,
    # decode_outputs shape => (1, total_obj_nums, 85)
    decode_outputs = torch.cat(total_layer_output, 1)

    return decode_outputs


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, classes_num, obj_conf_thresh=0.2, nms_iou_thresh=0.5):
    # (x, y, w, h) => (x1, y1, x2, y2)
    box_corner = torch.FloatTensor(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    # 修改prediction中（x, y, w, h）为 (x1, y1, x2, y2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # 利用置信度进行第一轮筛选,去掉目标置信度小于阈值的结果
        conf_mask = (image_pred[:, 4] >= obj_conf_thresh).squeeze()
        image_pred = image_pred[conf_mask]
        if not image_pred.size(0):
            continue

        # class confidence array, shape = (obj_num, classes_num)
        classes_conf = image_pred[:, 5:5 + classes_num]

        # 在列维度上获取，最大的类别概率，及其索引
        class_conf, class_pred = torch.max(classes_conf, 1, keepdim=True)

        # 在列维度上拼接结果：
        # (x1, y1, x2, y2, obj_conf）+（class_conf, class_pred) =>
        # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # detections shape = (obj_num, 7)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # 获取当前结果中的所有种类，unique()为取唯一值
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        # 遍历每一个类别，分别进行NMS
        for c in unique_labels:
            # 获得某一类初步筛选后全部的预测结果
            detections_class = detections[detections[:, -1] == c]

            # 按照存在目标的置信度降序排列，得到排序后的置信度与索引(相对于原始结果的索引)
            confidence = detections_class[:, 4]
            _, conf_sort_index = torch.sort(confidence, descending=True)

            # 根据排序后的索引，重新调整结果，得到降序排列后的结果
            detections_class = detections_class[conf_sort_index]

            # 进行非极大值抑制
            max_detections = []
            # detections_class.size(0) == 0为止
            while detections_class.size(0):
                # 获取detections_class中置信度最高的，并扩充维度（便于后续进行IOU计算）
                max_detections.append(detections_class[0].unsqueeze(0))

                # 如果只有一个结果，则没必要进行后续的比较
                if len(detections_class) == 1:
                    break

                # 计算末尾最高置信度的box与其它置信度的box的IOU，得到(detections_class.size - 1)个iou值
                ious = bbox_iou(max_detections[-1], detections_class[1:])

                # 只保留小于IOU阈值的目标，注意要将最大置信度的box剔除
                detections_class = detections_class[1:][ious < nms_iou_thresh]

            # 将非极大值抑制后的结果拼接(堆叠)
            max_detections = torch.cat(max_detections).data

            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # 将检测框的坐标还原至原始图像
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def draw_result(CLASSES, detections=None, image_src=None, input_size=(640, 640), line_thickness=None,
                text_bg_alpha=0.0):
    if detections is None:
        return

    labels = detections[..., -1]
    boxs = detections[..., :4]
    confs = detections[..., 4]

    h, w, c = image_src.shape

    boxs[:, :] = scale_coords(input_size, boxs[:, :], (h, w)).round()

    tl = line_thickness or round(0.002 * (w + h) / 2) + 1
    for i, box in enumerate(boxs):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1.numpy()), int(y1.numpy()), int(x2.numpy()), int(y2.numpy())
        np.random.seed(int(labels[i].numpy()) + 2020)
        color = (np.random.randint(0, 255), 0, np.random.randint(0, 255))
        cv2.rectangle(image_src, (x1, y1), (x2, y2), color, max(int((w + h) / 600), 1), cv2.LINE_AA)
        conf = '{0:.3f}'.format(confs[i])
        label = CLASSES[int(labels[i].numpy())] + "|" + conf
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = x1 + t_size[0] + 3, y1 - t_size[1] - 5
        if text_bg_alpha == 0.0:
            cv2.rectangle(image_src, (x1 - 1, y1), c2, color, cv2.FILLED, cv2.LINE_AA)
        else:
            # 透明文本背景
            alphaReserve = text_bg_alpha  # 0：不透明 1：透明
            BChannel, GChannel, RChannel = color
            xMin, yMin = int(x1 - 1), int(y1 - t_size[1] - 3)
            xMax, yMax = int(x1 + t_size[0]), int(y1)
            image_src[yMin:yMax, xMin:xMax, 0] = image_src[yMin:yMax, xMin:xMax, 0] * alphaReserve + BChannel * (
                        1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 1] = image_src[yMin:yMax, xMin:xMax, 1] * alphaReserve + GChannel * (
                        1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 2] = image_src[yMin:yMax, xMin:xMax, 2] * alphaReserve + RChannel * (
                        1 - alphaReserve)
        cv2.putText(image_src, label, (x1 + 3, y1 - 4), 0, tl / 3, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)