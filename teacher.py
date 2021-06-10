from models.yolo_distill import Model
from utils.torch_utils import select_device, intersect_dicts
from utils.general import non_max_suppression, scale_coords, xyxy2xywh

import numpy as np
import torch


class TeacherModel(object):
    def __init__(self, conf_thres=0.5, iou_thres=0.3, imgsz=640):
        self.model = None
        self.device = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

    def init_model(self, weights, device, batch_size, nc):

        self.device = device

        # load checkpoint
        ckpt = torch.load(weights, map_location=self.device)
        self.model = Model(ckpt['model'].yaml, ch=3, nc=nc).to(
            self.device)  # create
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(
            state_dict, self.model.state_dict(), exclude=['anchor'])  # intersect
        self.model.load_state_dict(state_dict, strict=False)  # load
        self.model.eval()
        self.stride = int(self.model.stride.max())
        self.nc = nc

    def __call__(self, imgs, tar_size=[640, 640]):
        targets = []
        with torch.no_grad():
            preds = self.model(imgs)[0]
            for img_id in range(imgs.shape[0]):

                pred = preds[img_id:img_id+1]
                pred = non_max_suppression(
                    pred, self.conf_thres, self.iou_thres, distill=True, agnostic=False)

                for det in pred:  # detections per image
                    gn = torch.tensor(tar_size)[[1, 0, 1, 0]]
                    if len(det):
                        # Rescale boxes from img_size to img0 size
                        det[:, :4] = scale_coords(
                            imgs[img_id].unsqueeze(0).shape[2:], det[:, :4], tar_size).round()

                        for value in reversed(det):
                            xyxy, cls_id = value[:4], value[5]
                            logits = value[-self.nc:].tolist()
                            xywh = (xyxy2xywh(torch.tensor(xyxy.cpu()).view(1, 4)
                                              ) / gn).view(-1).tolist()  # normalized xywh
                            line = [img_id, int(cls_id)]
                            line.extend(xywh)
                            line.extend(logits)
                            targets.append(line)

            return torch.tensor(np.array(targets), dtype=torch.float32)


if __name__ == '__main__':

    import cv2
    from utils.torch_utils import select_device

    teacher = TeacherModel(conf_thres=0.01)

    teacher.init_model('weights/yolov5s.pt', '0', 1, 20, 'models/yolov5l.yaml')

    # img0 = cv2.imread('../xingren.jpg')
    # img0, bboxes = teacher.predict(img0)
    # cv2.imshow('winname', img0)
    # cv2.waitKey(0)

    imgs = torch.rand((2, 3, 640, 640)).to(teacher.device)
    targets = teacher.generate_batch_targets(imgs)
