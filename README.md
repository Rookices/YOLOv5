本工程在https://github.com/ultralytics/yolov5/releases/tag/v4.0 的基础上适配海思平台，详细优化清单见文末todo list

----

##### 1.环境配置

Python>=3.7 and pytorch>=1.7

pip install -r requirements.txt

------------------

##### 2.数据集制作

支持txt图库下载，由参数args.urlPath传入，并完成数据集划分及格式转换，第一次下载图片时查看路径下是否有之前下载的图片，如不需要则删除，只第一次训练时需下载图片。默认训练时加载1.txt，测试时加载2.txt。  
数据集树状结构为

```
yolov5
├── data   
│   ├── images   存放图片
│   │   ├── train
│   │   ├── val
│   │   └── test
│   └── labels   存放标签
│       ├── train
│       ├── val
│       └── test
└── others
```

----

##### 3.配置文件

1. 超参(--hyp)： ./data/hyp.scratch.yaml   (可自行修改超参，如学习率，数据增强所用的超参等)
2. 模型配置(--cfg)： ./models/yolov5s.yaml  (为方便海思移植，已将focus层替换为conv层，upsample层替换为deconv层，且deconv层支持yolo缩放)
3. 数据集(--data)： ./data/ab.yaml   (根据自己数据集**更改图片路径**，类别及类别数)
4. 其他模型保存策略(--f1_factor)：新增基于f1评价指标的保存策略；>1时，R权重上升，反之越接近0，P权重越高
P.S. 在训练和测试阶段，无特殊情况，配置文件均为上述默认设置。
---

##### 4.训练（含剪枝：步骤2-4）

1. 原始模型训练

   ```
   python train.py --weights ./weights/yolov5s.pt --epochs 1000 
   ```

2. 稀疏训练
   ```
   python train.py --sl_factor <设定稀疏率，建议6e-4> --weights <半精度模型原始模型> --epochs 500
   ```

3. 剪枝，剪枝模型路径下会自动生成剪枝稀疏度分布图
   ```
   python pruning.py --weights <稀疏后模型> --save_path <剪枝模型保存路径> --thres <默认剪枝阈值0.01>
   ```

4. 模型微调

   ```
   python train.py --weights <剪枝后模型> --epochs 500 --ft_pruned
   ```

5. 冻结训练，新增数据集进行二次训练(缩短训练时间，可能牺牲少量精度)

   ```
   freeze = ['model.%s.' % x for x in range(10)]  # 修改train.py后按4.1训练；其中对于s模型，0-9为backbone
   ```
   Tips：中断训练后可在opt.yaml中修改配置，通过 "python train.py --resume" 实现继续训练

6. 模型量化(fp16→fp8)
   ```
   python slim.py --in_weights <原模型> --out_weights <输出> --device cpu
   ```
---

##### 5.测试

1. 图片推理
   ```
   python detect.py --source <图片路径> --weights <模型> --conf-thres <置信度阈值默认0.25> --iou-thres <IoU阈值默认0.5>
   ```
2. 模型评估(含FLOPs及耗时)
   ``` 
   python test.py --source <图片路径> --weights <模型>  --conf-thres <置信度阈值默认0.25> --iou-thres <IoU阈值默认0.5> --task <选择数据集val or test>
   ```
   P.S. 更改ab.yaml中的test路径可实现测试集更换
   
3. 批量导出稀疏度分布图(自动识别last前缀模型，导出路径为模型根目录)
   ```
   python img_bn_analyze.py --weights <模型所在根目录>
   ```
---

##### 6.自动化工具

1. 生成预测标签. 即在第5.1基础上将--save-txt, --save-conf 设置为True；会将预测.txt额外保存至./input/detection-results/
   ```
   python detect.py --source <图片路径> --weights <模型> --conf-thres <置信度阈值默认0.25> --iou-thres <IoU阈值默认0.5> --save-txt --save-conf
   ```
   
2. 生成真实标签，save_path为./input/ground-truth/
   ```
   python get_gt_txt.py --testPath <测试集图片路径>
   ```
   
3. 自动计算map，结果保存在result文件夹下

   ```
   python auto_analysis.py --testPath <测试集图片路径>
   ```

4. 新增样本自动标注，自动md5重命名，权重文件路径要在./inference/load_weights.py中修改

   ```
   python auto_label.py --img-path <新样本路径>  --save-pred <是否保存预测图片>  --img-size <图片尺寸，默认640>
   ```

---

##### 7.模型转换(支持onnx&caffe及海思prototxt导出)  

1. yolo2onnx

   python models/export.py --weights <需转换模型> --img-size <训练时图片输入尺寸>

2. onnx简化
   
   python -m onnxsim <onnx模型> <简化模型保存路径>
   
3. onnx2caffe   
   
   1）修改 ./yolov5_onnxcaffe/convertCaffe.py
   ```
        onnx_path         <onnx简化模型>
        prototxt_path     <prototxt保存路径>
        hisiprototxt_path <hisi_prototxt保存路径>
        caffemodel_path   <caffemodel保存路径>
   ```
   
   2）python yolov5_onnxcaffe/convertCaffe.py
---

##### todo list

###### ⚪训练相关

- [x] 支持多个训练样本txt导入，指定测试样本2.txt
- [x] 支持数据集划分，对每个txt单独按排序划分
- [x] 支持新增样本自动标注**auto_label.py**
- [x] 支持最后几个epochs关闭数据增强进行微调，调用**--remove_aug <最后几个epochs>**
- [x] 新增anchors保存策略，保存至checkpoint和txt中
- [x] 新增cutmix数据增强，需开启mosaic，不建议与mixup同时开启

###### ⚪模型压缩

- [x] 支持基于稀疏训练的通道剪枝，新增余弦梯度下降
- [x] 支持剪枝后网络结构自定义，以及通道自动8倍补齐
- [ ] 支持知识蒸馏

###### ⚪模型转换

- [x] 新增模型量化脚本，支持对中断训练的模型进行半精度转换
- [x] 支持自动导出海思prototxt

