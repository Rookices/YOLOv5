import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
import copy
import matplotlib.pyplot as plt
from models.yolo import Model
from utils.torch_utils import intersect_dicts


def load_model(weights):
    ckpt = torch.load(weights, map_location=device)  # load checkpoint
    model = Model(ckpt['model'].yaml).to(device)  # create
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
    model.load_state_dict(state_dict, strict=False)  # load
    print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    assert len(state_dict) == len(model.state_dict())

    model.float()
    model.model[-1].export = True
    return model

def bn_analyze(prunable_modules, save_path=None):
    bn_val = []
    max_val = []
    for layer_to_prune in prunable_modules:
        # select a layer
        weight = layer_to_prune.weight.data.detach().cpu().numpy()
        max_val.append(max(weight))
        bn_val.extend(weight)
    bn_val = np.abs(bn_val)
    max_val = np.abs(max_val)
    bn_val = sorted(bn_val)
    max_val = sorted(max_val)
    plt.hist(bn_val,bins=101, align="mid", log=True, range=(0, 1.0))
    if save_path is not None:
        if os.path.isfile(save_path):
            os.remove(save_path)
        plt.savefig(save_path)
    return bn_val, max_val

def bn_img(ori_model, example_inputs, output_transform):
    model = copy.deepcopy(ori_model)
    model.cpu().eval()

    prunable_module_type = (nn.BatchNorm2d)

    # ignore_idx = [230, 260, 290]

    prunable_modules = []
    for i, m in enumerate(model.modules()):
        # if i in ignore_idx:
        #     continue
        if isinstance(m, prunable_module_type):
            prunable_modules.append(m)
    ori_size = tp.utils.count_params(model)
    bn_val, max_val = bn_analyze(prunable_modules, os.path.splitext(opt.save_path)[0] + "_before_pruning.jpg")

    with torch.no_grad():
        out = model(example_inputs)
        out2 = ori_model(example_inputs)
        if output_transform:
            out = output_transform(out)
            out2 = output_transform(out2)
        # print("  Params: %s => %s" % (ori_size, tp.utils.count_params(model)))

    return model

def getFileName1(path,suffix):
    input_template_All=[]
    f_list = os.listdir(path)
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] ==suffix:
            if i.startswith('last'):
                # 筛选所有last模型
                input_template_All.append(i)
    return input_template_All


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="./bn/", type=str, help='模型所在根目录')
    parser.add_argument('--save_path', default="", type=str, help='')
    opt = parser.parse_args()

    a=getFileName1(opt.weights, ".pt")

    for i in a:
        weights = opt.weights+i
        save_dir = opt.save_path if os.path.isdir(opt.save_path) else os.path.dirname(os.path.abspath(weights))
        save_name = os.path.splitext(os.path.basename(weights))[0] + '.img'
        opt.save_path = os.path.join(save_dir, save_name)

        device = torch.device('cpu')
        model = load_model(weights)

        example_inputs = torch.zeros((1, 3, 640, 640), dtype=torch.float32).to()
        output_transform = None

        plt.title(i)
        bn_img(model, example_inputs=example_inputs,
                                     output_transform=output_transform)

        plt.clf()
        # print("Saved", opt.save_path)


