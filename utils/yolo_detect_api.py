from tools.auto_label.load_weights import model_file
from utils.datasets import *
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box


class yolo_inference():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

    def transforms(self, image, new_shape):
        """
        预处理
        :param image:原图
        :param new_shape: 图片尺寸
        :return: 处理后的图片
        """
        # Padded resize
        img = letterbox(image, new_shape)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def Process(self, pred, img, image, names, colors):
        """
        结果记录
        :param pred:经过nms的预测框
        :param img: 经过预处理的图片
        :param image: 预测结果图
        :param names: 标签名
        :param colors: 预测框颜色
        :return: 预测结果图，位置及标签信息，roi区域
        """
        image_copy = image.copy()
        xyxy_name = []  # xyxy and label_name
        img_crop = []
        h, w, _ = image_copy.shape
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                # Write results
                det = det[det[:, 0].argsort(descending=False)]
                # for *xyxy, conf, cls in det:
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=3)
                    img_crop.append(image_copy[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])])
                    xyxy_name.append([str(int(xyxy[0])), str(int(xyxy[1])), str(int(xyxy[2])), str(int(xyxy[3])), names[int(cls)]])
        return image, xyxy_name, img_crop

    def detect(self, model_file, image, new_shape, conf_thres=0.3, iou_thres=0.3):
        """
        前向推理
        :param model_file: 模型文件
        :param image:图片
        :param new_shape: 图片尺寸
        :param conf_thres: 置信度阈值
        :param iou_thres: iou阈值
        :return: Process()
        """
        # Load model
        model = torch.load(model_file, map_location=self.device)['model'].float().eval()  # load FP32 model
        if self.half:
            model.half()  # to FP16

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names  # class name
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]  # box colors

        # Run inference
        img = self.transforms(image, new_shape)
        with torch.no_grad():
            pred = model(img, augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        return self.Process(pred, img, image, names, colors)
